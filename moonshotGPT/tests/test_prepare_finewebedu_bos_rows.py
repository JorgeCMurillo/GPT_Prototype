import json
from concurrent.futures import ThreadPoolExecutor

from research.bos_aligned_proto.data.prepare_finewebedu_bos_rows import (
    PackingStats,
    ProgressMonitor,
    atomic_write_bytes,
)


class _DummyProgress:
    def __init__(self) -> None:
        self.total = 0
        self.postfix = ""
        self.closed = False

    def update(self, value: int) -> None:
        self.total += int(value)

    def set_postfix_str(self, value: str, refresh: bool = False) -> None:
        self.postfix = value

    def close(self) -> None:
        self.closed = True


class _DummyPrefetcher:
    pending_batches = 0
    queue_capacity = 4


class _DummyWriter:
    pending_writes = 0
    queue_capacity = 2


class _DummySink:
    current_shard_idx = 3
    current_shard_fill_rows = 7
    current_shard_fill_tokens = 77
    total_tokens_written = 1234


def test_atomic_write_bytes_is_safe_under_concurrent_same_path(tmp_path) -> None:
    path = tmp_path / "packing_status.json"

    def worker(worker_idx: int) -> None:
        for write_idx in range(100):
            payload = f"worker={worker_idx} write={write_idx}".encode("utf-8")
            atomic_write_bytes(str(path), payload)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, idx) for idx in range(8)]
        for future in futures:
            future.result()

    assert path.exists()
    assert path.read_bytes()


def test_progress_monitor_concurrent_refreshes_do_not_crash(tmp_path) -> None:
    progress = _DummyProgress()
    monitor = ProgressMonitor(
        progress=progress,
        out_dir=str(tmp_path),
        buffer_docs=16,
        prefetcher=_DummyPrefetcher(),
        writer=_DummyWriter(),
        sink=_DummySink(),
        stats=PackingStats(docs_processed=25, tokens_cropped_total=5),
        metrics=("docs", "crop", "pre", "wr"),
        refresh_interval_seconds=0.0,
        resume_from_shard=3,
        active_resume_state_shard_idx=3,
    )

    def source_updates() -> None:
        for docs_seen in range(1, 150):
            monitor.set_source_progress(
                docs_seen_total=docs_seen,
                replay_target_docs=250,
                replay_complete=(docs_seen >= 250),
            )
            monitor.set_doc_buffer_len(docs_seen % 16)

    def checkpoint_updates() -> None:
        for shard_idx in range(3, 150):
            monitor.on_tokens_written(1024)
            monitor.on_shard_checkpoint_saved(shard_idx)

    monitor.emit_startup_summary()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(source_updates),
            executor.submit(checkpoint_updates),
        ]
        for future in futures:
            future.result()
    monitor.mark_completed()
    monitor.close()

    status_path = tmp_path / "packing_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "completed"
    assert payload["active_resume_state_shard_idx"] >= 3
    assert "explanation" in payload
    assert progress.closed is True
