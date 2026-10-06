"""Device selection and GPU introspection. torch imported lazily so the
Streamlit app (which doesn't ship torch) can still import src.utils.*.
"""
from __future__ import annotations


def _torch():
    import torch
    return torch


def get_device(prefer: str = "auto") -> str:
    if prefer == "cpu":
        return "cpu"
    try:
        torch = _torch()
    except ImportError:
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_info() -> dict:
    try:
        torch = _torch()
    except ImportError:
        return {"torch": None, "cuda_available": False}
    info = {"torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info


def amp_dtype():
    try:
        torch = _torch()
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


_TABLE = {
    "sbert":      {8: 64, 16: 128, 24: 256},
    "bert_base":  {8: 8,  16: 16,  24: 32},
    "modernbert": {8: 4,  16: 8,   24: 16},
}


def suggested_batch_size(vram_gb: float, task: str) -> int:
    bucket = 8 if vram_gb <= 8 else (16 if vram_gb <= 16 else 24)
    return _TABLE.get(task, _TABLE["sbert"])[bucket]
