from concurrent.futures import ThreadPoolExecutor
import math

import librosa
import numpy as np
from numba import jit


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def split_lr_waves(
    wave, mid_side=False, mid_side_b2=False, reverse=False
):
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.add(wave[0], wave[1]) / 2
        wave_right = np.subtract(wave[0], wave[1])
    elif mid_side_b2:
        wave_left = np.add(wave[1], wave[0] * 0.5)
        wave_right = np.subtract(wave[0], wave[1] * 0.5)
    else:
        wave_left = wave[0]
        wave_right = wave[1]

    return wave_left, wave_right


def run_librosa_stft(wv, n_fft, hop_length, reverse):
    if reverse:
        return librosa.stft(wv, n_fft=n_fft, hop_length=hop_length)
    return librosa.stft(np.asfortranarray(wv), n_fft=n_fft, hop_length=hop_length)

def wave_to_spectrogram_mt(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):

    with ThreadPoolExecutor(max_workers=2) as tp:
        spec = np.asfortranarray(
            [spec for spec in tp.map(
                run_librosa_stft,
                split_lr_waves(wave, mid_side, mid_side_b2, reverse),
                [n_fft, n_fft], [hop_length, hop_length], [reverse, reverse]
            )]
        )

    return spec


def combine_spectrograms(specs, mp):
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter
    if (
        mp.param["pre_filter_start"] > 0
    ):  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
        if bands_n == 1:
            spec_c = fft_lp_filter(
                spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
            )
        else:
            gp = 1
            for b in range(
                mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
            ):
                g = math.pow(
                    10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                )
                gp = g
                spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


@jit(nopython=True)
def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[
                :, :, s + fade_size : e - fade_size
            ]
            old_e = e

    return mag


def run_librosa_istft(specx, hop_length):
    return librosa.istft(np.asfortranarray(specx), hop_length=hop_length)

def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):

    with ThreadPoolExecutor(max_workers=2) as tp:
        wave_left, wave_right = tp.map(run_librosa_istft, spec, [hop_length, hop_length])

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray(
            [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
        )
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.ndarray(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                wave = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type="sinc_fastest",
                )
            else:  # mid
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
                # wave = librosa.core.resample(wave2, bp['sr'], sr, res_type="sinc_fastest")
                wave = librosa.resample(
                    wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy"
                )

    return wave.T


@jit(nopython=True)
def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


@jit(nopython=True)
def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, pre_filter_start):
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    pre_filter_start
                    - 10
                    - input_high_end.shape[1] : pre_filter_start
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    pre_filter_start
                    - 10
                    - input_high_end.shape[1] : pre_filter_start
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)
