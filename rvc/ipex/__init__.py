try:
    import torch
    if torch.xpu.is_available():
        from .init import ipex_init
        ipex_init()
        from .gradscaler import gradscaler_init
except Exception:  # pylint: disable=broad-exception-caught
    pass
