#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = [
    ("ttft_s_mean", "TTFT mean (s)"),
    ("tpot_s_mean", "TPOT mean (ms)"),
    ("end_to_end_latency_s_mean", "E2E mean (s)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize baseline/HBM/UCM llmperf results."
    )
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args()


def load_payload(results_dir: Path, scenario: str) -> dict[str, Any]:
    with open(results_dir / f"{scenario}_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def improve_latency(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline == 0:
        return None
    return (baseline - value) / baseline * 100.0


def fmt(value: float | None, scale: float = 1.0) -> str:
    if value is None:
        return "n/a"
    return f"{value * scale:.4f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    payloads = {
        name: load_payload(results_dir, name) for name in ["baseline", "hbm", "ucm"]
    }
    baseline_metrics = payloads["baseline"]["metrics"]

    summary: dict[str, Any] = {
        "results_dir": str(results_dir),
        "scenarios": {},
    }

    rows = [
        "| Scenario | TTFT mean (s) | TTFT improvement | TPOT mean (ms) | TPOT improvement | E2E mean (s) | E2E improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    labels = {
        "baseline": "Baseline",
        "hbm": f"HBM prefix cache {payloads['hbm']['hit_rate']}% hit",
        "ucm": f"UCM external cache {payloads['ucm']['hit_rate']}% hit",
    }

    for scenario in ["baseline", "hbm", "ucm"]:
        metrics = payloads[scenario]["metrics"]
        scenario_summary = {}
        for key, _label in METRICS:
            scenario_summary[key] = metrics.get(key)
            scenario_summary[f"{key}_improvement_vs_baseline_pct"] = improve_latency(
                baseline_metrics.get(key), metrics.get(key)
            )
        summary["scenarios"][scenario] = scenario_summary

        ttft = metrics.get("ttft_s_mean")
        tpot = metrics.get("tpot_s_mean")
        e2e = metrics.get("end_to_end_latency_s_mean")
        ttft_imp = improve_latency(baseline_metrics.get("ttft_s_mean"), ttft)
        tpot_imp = improve_latency(baseline_metrics.get("tpot_s_mean"), tpot)
        e2e_imp = improve_latency(
            baseline_metrics.get("end_to_end_latency_s_mean"), e2e
        )
        if scenario == "baseline":
            ttft_imp = tpot_imp = e2e_imp = 0.0
        rows.append(
            "| "
            + " | ".join(
                [
                    labels[scenario],
                    fmt(ttft),
                    fmt_pct(ttft_imp),
                    fmt(tpot, scale=1000.0),
                    fmt_pct(tpot_imp),
                    fmt(e2e),
                    fmt_pct(e2e_imp),
                ]
            )
            + " |"
        )

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
        f.write("\n")

    print("\n".join(rows))
    print(f"[INFO] Wrote {results_dir / 'summary.json'}")
    print(f"[INFO] Wrote {results_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
