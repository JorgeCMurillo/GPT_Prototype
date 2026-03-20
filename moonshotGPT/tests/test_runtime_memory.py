import torch

import runtime_memory


def test_format_memory_usage_postfix_reports_ram(monkeypatch) -> None:
    monkeypatch.setattr(runtime_memory, "get_process_rss_bytes", lambda: 3 * 1024 ** 3)
    result = runtime_memory.format_memory_usage_postfix(torch.device("cpu"))
    assert result == "ram=3.0GiB"


def test_format_memory_usage_postfix_reports_cuda_and_ram(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 2 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 3 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 4 * 1024 ** 3)
    monkeypatch.setattr(runtime_memory, "get_process_rss_bytes", lambda: 5 * 1024 ** 3)

    result = runtime_memory.format_memory_usage_postfix(torch.device("cuda", 0))

    assert result == "vram=2.0GiB/3.0GiB peak=4.0GiB ram=5.0GiB"
