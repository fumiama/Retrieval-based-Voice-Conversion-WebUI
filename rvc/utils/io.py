import os
from typing import Union, IO
import torch

if hasattr(torch.serialization, "FILE_LIKE"):
    FileLike = torch.serialization.FILE_LIKE
elif hasattr(torch.types, "FILE_LIKE"):
    FileLike = torch.types.FileLike
else:
    FileLike = Union[str, os.PathLike, IO[bytes]]
