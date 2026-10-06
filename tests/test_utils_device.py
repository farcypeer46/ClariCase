from src.utils.device import get_device, gpu_info, suggested_batch_size


def test_get_device_returns_cpu_when_preferred():
    assert get_device(prefer="cpu") == "cpu"


def test_gpu_info_returns_dict_with_versions():
    info = gpu_info()
    assert set(info) >= {"torch", "cuda_available"}


def test_suggested_batch_size_returns_positive_int():
    assert suggested_batch_size(vram_gb=8, task="sbert") > 0
    assert suggested_batch_size(vram_gb=24, task="sbert") \
           >= suggested_batch_size(vram_gb=8, task="sbert")
