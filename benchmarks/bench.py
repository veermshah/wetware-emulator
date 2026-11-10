"""Benchmark simulation speed across network sizes and neuron models.

python benchmarks/bench.py            # full suite, prints a markdown table
python benchmarks/bench.py --quick    # 1000 neurons only
"""

import argparse

from wetware.benchmark import format_table, machine, measure, run_suite

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--duration-ms", type=float, default=5000.0)
    a = ap.parse_args()
    rows = (
        [measure(1000, duration_ms=a.duration_ms)]
        if a.quick
        else run_suite(duration_ms=a.duration_ms)
    )
    print(machine())
    print(format_table(rows))
