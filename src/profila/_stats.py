"""
Aggregate call stacks.
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
@dataclass(frozen=True)
class Frame:
    file: str
    line: int
    name: str = ""



@dataclass(frozen=True)
class FinalStats:
    """
    Final result when all stats are gathered.
    """

    total_samples: int
    percent_bad_samples: float
    percent_other_samples: float
    # Map path to mapping of line number to percentage.
    numba_samples: dict[str, dict[int, float]]

    def total_percent(self) -> float:
        """
        Add up all percentages, should always be approximately 100%.
        """
        result = self.percent_bad_samples + self.percent_other_samples
        for line_counts in self.numba_samples.values():
            result += sum(line_counts.values())
        if result == 0.0:
            # Got absolutely nothing, which is also valid!
            result = 100.0
        return result


@dataclass
class Stats:
    # Map Python filenames to per-line counts. Should be Numba samples only.
    path_to_line_counts: defaultdict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Full stack trace samples count (tuple of Frame)
    stack_counts: Counter[tuple[Frame, ...]] = field(default_factory=Counter)
    # Samples we couldn't parse:
    bad_samples: int = 0
    # Samples that weren't Numba based:
    other_samples: int = 0

    def total_samples(self) -> int:
        """
        Total number of all samples.
        """
        result = self.bad_samples + self.other_samples
        for line_counts in self.path_to_line_counts.values():
            result += sum(line_counts.values())
        return result

    def add_sample(self, sample: Optional[list[Frame]]) -> None:
        """
        Add a sample.
        """
        if sample is None:
            self.bad_samples += 1
            return

        self.stack_counts[tuple(sample)] += 1

        for frame in sample:
            if frame.file.endswith(".py"):
                self.path_to_line_counts[frame.file][frame.line] += 1
                return

        self.other_samples += 1

    def to_folded_stacks(self) -> str:
        """
        Generate folded stack strings (compatible with FlameGraph generators).
        Each line format: frame1;frame2;frame3 count
        """
        lines = []
        if self.bad_samples > 0:
            lines.append(f"[bad_sample] {self.bad_samples}")
        if self.other_samples > 0:
            lines.append(f"[non_numba] {self.other_samples}")

        for stack, count in self.stack_counts.items():
            frame_strs = [
                f"{f.name} ({os.path.basename(f.file)}:{f.line})" if f.name else f"{os.path.basename(f.file)}:{f.line}"
                for f in stack
            ]
            lines.append(f"{';'.join(frame_strs)} {count}")

        return "\n".join(lines)

    def finalize(self) -> FinalStats:
        """
        Calculate final stats for human rendering.
        """
        total_samples = self.total_samples()

        def to_percent(count: int) -> float:
            if total_samples == 0:
                return 0.0
            return round((count / total_samples) * 100, 1)

        percent_bad_samples = to_percent(self.bad_samples)
        percent_other_samples = to_percent(self.other_samples)
        numba_samples = {}
        for filename, counts in self.path_to_line_counts.items():
            filename_counts: dict[int, float] = {}
            numba_samples[filename] = filename_counts
            for line_number, count in counts.items():
                filename_counts[line_number] = to_percent(count)

        final_stats = FinalStats(
            total_samples=total_samples,
            percent_bad_samples=percent_bad_samples,
            percent_other_samples=percent_other_samples,
            numba_samples=numba_samples,
        )
        assert -5.0 < final_stats.total_percent() - 100 < 5.0
        return final_stats
