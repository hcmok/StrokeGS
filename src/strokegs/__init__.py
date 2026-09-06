from pathlib import Path

from torch.utils.cpp_extension import load

EXTENSION_DIR = Path(__file__).parent.resolve()
BUILD_DIR = EXTENSION_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)

sources = [
    str(EXTENSION_DIR / "csrc" / "rasterize.cpp"),
    str(EXTENSION_DIR / "csrc" / "cuda" / "rasterize.cu"),
]

strokegs = load(
    name="_C",
    sources=sources,
    build_directory=BUILD_DIR,
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    verbose=True,
)
from . import ops
