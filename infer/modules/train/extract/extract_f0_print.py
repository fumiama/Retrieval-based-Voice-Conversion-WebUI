import os
import sys

import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger("rvc.F0Print")

from rvc.f0 import F0Predictor, CRePE, PM, Dio, Harvest, RMVPE

from multiprocessing import Process

# exp_dir = sys.argv[1]
# f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

def printt(strr):
    logger.info(strr)

def write_f0(predictor: F0Predictor, inp_path: str, coarse_path: str, feature_path: str) -> None:
    try:
        out_features = predictor.compute_f0(inp_path)
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
    if type(predictor) == type(RMVPE):
        predictor = predictor("assets/rmvpe/rmvpe.pt", is_half, device)
    else:
        predictor = predictor(device=device)
    featureInput = predictor

    # TODO: rename these vars to not be confusing
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
        opt_path1 = f"{coarse_path}/{name}"
        opt_path2 = f"{feature_path}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    ps = []
    for idx, (inp_path, coarse_path, feature_path) in enumerate(paths):
        p = Process(target=write_f0, args=(featureInput, inp_path, coarse_path, feature_path))
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
