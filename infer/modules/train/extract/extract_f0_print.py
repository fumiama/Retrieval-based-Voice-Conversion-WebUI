import os
import sys

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

def extract_features(predictor: F0Predictor, expected_dir: str, is_half: bool, device: str) -> None:
    """
    Extract features

    Args:
        expected_dir: str - Directory to store extracted features
        cores: int - Number of CPU cores to use
        method_f0: str - F0 method to use { pm, harvest, dio, rmvpe }
    """
    featureInput = predictor(device=device) if not isinstance(predictor, RMVPE) else predictor(is_half=is_half, device=device)

    # TODO: rename these vars to not be confusing
    inp_root = f"{expected_dir}/1_16k_wavs"
    opt_root1 = f"{expected_dir}/2a_f0"
    opt_root2 = f"{expected_dir}/2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    paths = []
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    ps = []
    for path in paths:
        p = Process(target=featureInput.compute_f0, args=(path))
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
