"""
Reusable Live Progress Logger for Antigravity Entity Resolution Pipeline.
Provides clean, un-faked, real-time progress logging and heartbeats across major stages.
"""

import time

def format_time(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds == float('inf'):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class ProgressLogger:
    """
    Reusable logger that outputs live progress every N seconds or M percent completed.
    """
    def __init__(self, min_interval_sec: float = 60.0, pct_step: float = 5.0):
        self.min_interval_sec = min_interval_sec
        self.pct_step = pct_step
        self.last_log_time = 0.0
        self.last_log_pct = -1.0
        self.stage_start_time = 0.0
        self.current_stage = ""
        self.stage_num = 0
        self.total_stages = 0

    def start_stage(self, stage_num: int, total_stages: int, stage_name: str):
        self.stage_num = stage_num
        self.total_stages = total_stages
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.last_log_time = self.stage_start_time
        self.last_log_pct = -1.0
        print(f"\n[STAGE {stage_num}/{total_stages}] {stage_name}...", flush=True)

    def should_log(self, current: int, total: int) -> bool:
        now = time.time()
        if total <= 0:
            return (now - self.last_log_time) >= self.min_interval_sec
        
        pct = (current / total) * 100.0
        if current >= total:
            return True
        if (now - self.last_log_time) >= self.min_interval_sec:
            return True
        if self.last_log_pct < 0 or (pct - self.last_log_pct) >= self.pct_step:
            return True
        return False

    def log_progress(
        self,
        current: int,
        total: int,
        candidates: int = None,
        matches: int = None,
        workers: int = None,
        batch_num: int = None,
        force: bool = False
    ):
        now = time.time()
        if not force and not self.should_log(current, total):
            return

        elapsed = now - self.stage_start_time
        pct = (current / total * 100.0) if total > 0 else 0.0
        
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            rem_items = max(0, total - current)
            eta = rem_items / rate
        else:
            eta = 0.0

        elapsed_str = format_time(elapsed)
        eta_str = format_time(eta)

        parts = [
            f"[STAGE {self.stage_num}/{self.total_stages}] {self.current_stage}",
            f"Progress: {current:,}/{total:,} ({pct:.1f}%)"
        ]
        if candidates is not None:
            parts.append(f"Candidates: {candidates:,}")
        if matches is not None:
            parts.append(f"Matches: {matches:,}")
        if workers is not None:
            parts.append(f"Workers: {workers}")
        if batch_num is not None:
            parts.append(f"Batch: {batch_num}")
        parts.append(f"Elapsed: {elapsed_str} | ETA: {eta_str}")

        print(" | ".join(parts), flush=True)
        self.last_log_time = now
        self.last_log_pct = pct

    def heartbeat(self, message: str = "Active"):
        now = time.time()
        if (now - self.last_log_time) >= self.min_interval_sec:
            elapsed_str = format_time(now - self.stage_start_time)
            print(f"[HEARTBEAT - STAGE {self.stage_num}/{self.total_stages}] {self.current_stage} | {message} | Elapsed: {elapsed_str}", flush=True)
            self.last_log_time = now

    def finish_stage(self, stage_name: str = None):
        now = time.time()
        elapsed = now - self.stage_start_time
        s_name = stage_name or self.current_stage
        print(f"[COMPLETED STAGE {self.stage_num}/{self.total_stages}] {s_name} in {format_time(elapsed)} ({elapsed:.2f}s)\n", flush=True)
