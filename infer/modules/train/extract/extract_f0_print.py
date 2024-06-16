import os
import sys

import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger("rvc.F0Print")

from rvc.f0 import F0Predictor, CRePE, PM, Dio, Harvest, RMVPE, FCPE

from infer.lib.audio import load_audio

from torch.multiprocessing import Process, set_start_method

set_start_method("spawn", force=True)


# A helper function to log in both the terminal and the logfile
def __log(logfile, data: str) -> None:
    logger.info(data)
    logfile.write(f"{data}\n")


def save_f0(
    predictor: F0Predictor,
    inp_path: str,
    coarse_path: str,
    feature_path: str,
    logfile: str,
) -> None:
    """
    Compute the F0 and save the results in a log file

    Args:
        inp_path: str - Input path for the audio
        coarse_path: str - Path to save f0 coarse to
        feature_path: str - Path to save f0 features to
        logfile: str - The main log file of the f0 extraction
    """

    try:
        # So fun fact:
        # As it turns out, you just can't pickle the TextIOWrappers, for some reason?
        # Why is that? Who knows! But this unfortunately means that we have to
        # Open the logfiles within each thread in order to have anything done here...
        # This makes the code I/O bound and there's not a lot to do about it, i think
        __logfile = open(logfile, "w")

        out_features = predictor.compute_f0(load_audio(inp_path, 16000))
        out_coarse = 0  # TODO: add coarse f0
        np.save(
            feature_path,
            out_features,
            allow_pickle=False,
        )
        # np.save(
        #     coarse_path,
        #     out_coarse,
        #     allow_pickle=False,
        # )
    except Exception as e:
        __log(__logfile, f"Failed to compute f0 for - {inp_path}: {e}")


def extract_features(
    predictor: F0Predictor, expected_dir: str, is_half: bool, device: str
) -> None:
    """
    Extract features

    Args:
        expected_dir: str - Directory to store extracted features
        cores: int - Number of CPU cores to use
        method_f0: str - F0 method to use { pm, harvest, dio, rmvpe }
    """
    if predictor is RMVPE:
        predictor = predictor("assets/rmvpe/rmvpe.pt", is_half, device)
    elif predictor is CRePE or predictor is FCPE:
        predictor = predictor(
            hop_length=512, f0_min=50, f0_max=1100, sampling_rate=16000, device=device
        )
    else:
        predictor = predictor(
            hop_length=512, f0_min=50, f0_max=1100, sampling_rate=16000
        )
    featureInput = predictor

    inp_root = f"{expected_dir}/1_16k_wavs"
    coarse_path = f"{expected_dir}/2a_f0"
    feature_path = f"{expected_dir}/2b-f0nsf"

    os.makedirs(coarse_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)

    paths = []
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        coarse_path = f"{coarse_path}/{name}"
        feature_path = f"{feature_path}/{name}"
        paths.append([inp_path, coarse_path, feature_path])

    ps = []

    log_file = open(f"{expected_dir}/extract_f0_feature.log", "w")

    n_paths = len(paths)
    if n_paths == 0:
        __log(log_file, "No f0 to do")
    else:
        __log(log_file, f"F0 to do: {n_paths}")

    n = max(n_paths // 5, 1)  # 每个进程最多打印5条

    for idx, (inp_path, coarse_path, feature_path) in enumerate(paths):
        if idx % n == 0:
            __log(
                log_file, f"Computing f0; Current: {idx}; Total: {n_paths}; {inp_path}"
            )
        p = Process(
            name=f"extract_f0_feature_{idx}",
            target=save_f0,
            args=(
                featureInput,
                inp_path,
                coarse_path,
                feature_path,
                f"{expected_dir}/extract_f0_feature.log",
            ),
        )
        logger.debug(f"Starting extract_f0_feature_{idx} thread for {inp_path}")
        p.start()
        ps.append(p)

    for p in ps:
        p.join()


predictors = {
    "pm": PM,
    "harvest": Harvest,
    "dio": Dio,
    "rmvpe": RMVPE,
    "rmvpe_gpu": RMVPE,
    "crepe": CRePE,
    "fcpe": FCPE,
}


def call_extract_features(
    expected_dir: str, method_f0: str, is_half: bool, device: str
) -> None:
    """
    Extract features

    Args:
        expected_dir: str - Directory to store extracted features
        cores: int - Number of CPU cores to use
        method_f0: str - F0 method to use { pm, harvest, dio, rmvpe }
    """

    if method_f0 not in predictors:  # Check if the method is valid
        raise ValueError(f"Unknown feature extraction method: {method_f0}")

    extract_features(predictors[method_f0], expected_dir, is_half, device)
