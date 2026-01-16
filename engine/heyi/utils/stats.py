import time

import numpy as np


class ReqStats:
    def __init__(self):
        self._start_time = time.perf_counter()
        self._first_token_time = 0.0
        self._last_token_time = 0.0
        self._token_intervals = []

        self.prefill_ntokens = 0
        self.decode_ntokens = 0

        self.ttft = 0
        self.avg_tbt = 0
        self.p95_tbt = 0
        self.throughput = 0

    def on_prefill_done(self, ntokens: int):
        self.prefill_ntokens = ntokens
        self._first_token_time = time.perf_counter()
        self._last_token_time = self._first_token_time
        self.decode_ntokens = 1

    def on_decode1_done(self):
        current_time = time.perf_counter()
        self._token_intervals.append(current_time - self._last_token_time)
        self.decode_ntokens += 1
        self._last_token_time = current_time

    def summarize(self):
        ttft = (
            (self._first_token_time - self._start_time) * 1000
            if self._start_time and self._first_token_time
            else 0.0
        )

        if self._token_intervals:
            intervals = np.array(self._token_intervals)
            avg_tbt = np.mean(intervals) * 1000
            p95_tbt = np.percentile(intervals, 95) * 1000
        else:
            avg_tbt = p95_tbt = 0.0

        if self.decode_ntokens > 1:
            total_time = self._last_token_time - self._first_token_time
            throughput = (
                (self.decode_ntokens - 1) / total_time if total_time > 0 else 0.0
            )
        else:
            throughput = 0.0

        self.ttft=ttft
        self.avg_tbt=avg_tbt
        self.p95_tbt=p95_tbt
        self.throughput=throughput

        return dict(
            ttft=ttft,
            avg_tbt=avg_tbt,
            p95_tbt=p95_tbt,
            throughput=throughput,
        )

    def pretty_print_str(self) -> str:
        self.summarize()

        output = [
            "\n" + "=" * 60,
            "LLM Request Performance Statistics".center(60),
            "-" * 60,
            f"{'TTFT':<35}: {self.ttft:>8.2f} ms",
            f"{'Avg TBT':<35}: {self.avg_tbt:>8.2f} ms",
            f"{'P95 TBT':<35}: {self.p95_tbt:>8.2f} ms",
            f"{'Prefill Tokens':<35}: {self.prefill_ntokens:>8}",
            f"{'Generated Tokens':<35}: {self.decode_ntokens:>8}",
        ]

        output.append(f"{'Decode Throughput':<35}: {self.throughput:>8.2f} tok/s")
        output.append("=" * 60)
        return "\n".join(output)


if __name__ == "__main__":
    stats = ReqStats()
    time.sleep(0.12)
    stats.on_prefill_done(256)

    delays = [0.05, 0.048, 0.052, 0.049, 0.051, 0.055, 0.053, 0.047, 0.050, 0.052]
    for delay in delays:
        time.sleep(delay)
        stats.on_decode1_done()

    print(stats.pretty_print_str())
