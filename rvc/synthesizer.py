from collections import OrderedDict

import torch

from .layers.synthesizers import SynthesizerTrnMsNSFsid


def get_synthesizer(cpt: OrderedDict, device=torch.device("cpu")):
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        encoder_dim = 256
    elif version == "v2":
        encoder_dim = 768
    net_g = SynthesizerTrnMsNSFsid(
        *cpt["config"],
        encoder_dim=encoder_dim,
        use_f0=if_f0 == 1,
    )
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


def load_synthesizer(
    pth_path: torch.serialization.FILE_LIKE, device=torch.device("cpu")):
    return get_synthesizer(
        torch.load(pth_path, map_location=torch.device("cpu")),
        device,
    )
