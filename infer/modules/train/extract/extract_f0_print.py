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

# exp_dir = sys.argv[1]
# f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

def printt(strr):
    logger.info(strr)

def save_f0(predictor: F0Predictor, inp_path: str, coarse_path: str, feature_path: str) -> None:
    try:
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
        printt("Failed to compute f0 for - %s: %s" % (inp_path, e))


def extract_features(predictor: F0Predictor, expected_dir: str, is_half: bool, device: str) -> None:
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
        predictor = predictor(device=device)
    else:
        predictor = predictor()
    featureInput = predictor

    inp_root = f"{expected_dir}/1_16k_wavs"
    coarse_path = f"{expected_dir}/2a_f0"
    feature_path = f"{expected_dir}/2b-f0nsf"

    os.makedirs(coarse_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)

    log_f0 = lambda f0: open(f"{expected_dir}/extract_f0_feature.log").write(f"{f0}\n")

    paths = []
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        coarse_path = f"{coarse_path}/{name}"
        feature_path = f"{feature_path}/{name}"
        paths.append([inp_path, coarse_path, feature_path])

    ps = []
    for idx, (inp_path, coarse_path, feature_path) in enumerate(paths):
        p = Process(name=f"extract_f0_feature_{idx}", target=save_f0, args=(featureInput, inp_path, coarse_path, feature_path))
        logger.info(f"Starting extract_f0_feature_{idx} thread for {inp_path}")
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


def call_extract_features(expected_dir: str, method_f0: str, is_half: bool, device: str) -> None:
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
