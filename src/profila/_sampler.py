"""
Sampling profiler for Python and Numba execution.

Periodically samples thread stack frames (at 1000 Hz / 1ms interval) using sys._current_frames()
and records clean call stacks into Stats.
"""

from collections import Counter
import os
import sys
import threading
import time
from typing import List, Optional

from ._dwarf import JITDWARFResolver
from ._stats import Frame, Stats
from ._func_resolver import resolve_function_name

IGNORE_ROOT_FILENAMES = {
    "<frozen runpy>",
    "runpy.py",
    "profila/__main__.py",
    "_sampler.py",
}


class Sampler:
    """
    High-frequency thread stack sampler (1ms interval / 1000Hz).
    """

    def __init__(self, interval_seconds: float = 0.001) -> None:
        self.interval = interval_seconds
        self.stats = Stats()
        self.resolver = JITDWARFResolver()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._target_thread_id: Optional[int] = None

    def start(self, target_thread_id: Optional[int] = None) -> None:
        """
        Start sampling target thread (default: main thread).
        """
        if target_thread_id is None:
            target_thread_id = threading.main_thread().ident

        self._target_thread_id = target_thread_id
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Stats:
        """
        Stop sampling and return accumulated Stats.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.stats

    def _sample_loop(self) -> None:
        while self._running:
            start_time = time.perf_counter()
            self._take_sample()
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.0, self.interval - elapsed)
            time.sleep(sleep_time)

    def _take_sample(self) -> None:
        if self._target_thread_id is None:
            return

        frames_dict = sys._current_frames()
        frame = frames_dict.get(self._target_thread_id)

        if frame is None:
            self.stats.add_sample(None)
            return

        raw_stack: List[Frame] = []
        curr = frame
        while curr is not None:
            code = curr.f_code
            filename = os.path.abspath(code.co_filename)
            lineno = curr.f_lineno
            func_name = resolve_function_name(filename, lineno, code.co_name)
            raw_stack.append(Frame(file=filename, line=lineno, name=func_name))
            curr = curr.f_back

        if not raw_stack:
            self.stats.add_sample(None)
            return

        # Order from outer (root) to inner (leaf)
        raw_stack.reverse()

        # Trim profiler harness boilerplate frames from root
        clean_stack = [
            f for f in raw_stack
            if not any(ignore_name in f.file for ignore_name in IGNORE_ROOT_FILENAMES)
        ]

        if clean_stack:
            self.stats.add_sample(clean_stack)
        else:
            self.stats.add_sample(raw_stack)
