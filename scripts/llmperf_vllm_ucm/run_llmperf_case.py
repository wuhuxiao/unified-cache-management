#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "test"))

from common.llmperf.utils import common_metrics  # noqa: E402
from common.llmperf.utils.token_benchmark import run_token_benchmark  # noqa: E402
import common.llmperf.utils.openai_chat_completions_client as openai_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one project llmperf case against an OpenAI-compatible vLLM server."
    )
    parser.add_argument("--scenario", required=True, choices=["baseline", "hbm", "ucm"])
    parser.add_argument("--server-url", default="http://127.0.0.1:10035")
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "DeepSeek-V4-Flash"))
    parser.add_argument("--tokenizer-path", default=os.environ.get("MODEL_PATH", "/home/models/DeepSeek-V4-Flash"))
    parser.add_argument("--input-tokens", type=int, default=16000)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--hit-rate", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=int, default=60000)
    parser.add_argument("--request-timeout-s", type=int, default=1800)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "llmperf_vllm_ucm"))
    parser.add_argument("--prefill-sleep-s", type=float, default=2.0)
    return parser.parse_args()


def reset_prefix_cache(server_url: str) -> None:
    url = server_url.rstrip("/") + "/reset_prefix_cache"
    print(f"[INFO] Resetting vLLM prefix cache: {url}", flush=True)
    try:
        response = requests.post(url, timeout=10)
    except requests.RequestException as exc:
        print(f"[WARN] Failed to reset prefix cache: {exc}", flush=True)
        return

    if not 200 <= response.status_code < 300:
        print(
            f"[WARN] Prefix cache reset returned HTTP {response.status_code}: {response.text}",
            flush=True,
        )


def run_request(
    args: argparse.Namespace,
    phase: str,
    mean_input_tokens: int,
    mean_output_tokens: int,
) -> dict[str, Any]:
    return run_token_benchmark(
        llm_api="openai",
        model=args.model,
        test_timeout_s=args.timeout_s,
        max_num_completed_requests=args.requests,
        concurrent_requests=args.concurrency,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=0,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=0,
        results_dir=args.results_dir,
        random_seed=args.random_seed,
        openai_api_base=args.server_url.rstrip("/") + "/v1",
        tokenizer_path=args.tokenizer_path,
        user_metadata={
            "scenario": args.scenario,
            "phase": phase,
            "hit_rate": args.hit_rate,
            "server_url": args.server_url,
        },
    )


def nested(summary: dict[str, Any], metric: str, field: str) -> float | None:
    value = summary.get(metric)
    if not isinstance(value, dict):
        return None
    leaf = value.get(field)
    if leaf is None:
        return None
    return float(leaf)


def quantile(summary: dict[str, Any], metric: str, name: str) -> float | None:
    value = summary.get(metric)
    if not isinstance(value, dict):
        return None
    quantiles = value.get("quantiles")
    if not isinstance(quantiles, dict):
        return None
    leaf = quantiles.get(name)
    if leaf is None:
        return None
    return float(leaf)


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results", summary)
    return {
        "ttft_s_mean": nested(results, common_metrics.TTFT, "mean"),
        "ttft_s_p50": quantile(results, common_metrics.TTFT, "p50"),
        "tpot_s_mean": nested(results, common_metrics.TPOT, "mean"),
        "tpot_s_p50": quantile(results, common_metrics.TPOT, "p50"),
        "end_to_end_latency_s_mean": nested(results, common_metrics.E2E_LAT, "mean"),
        "end_to_end_latency_s_p50": quantile(results, common_metrics.E2E_LAT, "p50"),
        "num_completed_requests": results.get(common_metrics.NUM_COMPLETED_REQUESTS),
        "error_rate": results.get(common_metrics.ERROR_RATE),
        "total_throughput": summary.get("total_throughput"),
    }


def main() -> int:
    args = parse_args()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost," + os.environ.get("NO_PROXY", "")
    openai_client.timeout = args.request_timeout_s
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.hit_rate > 0:
        prefill_input = int(args.input_tokens * args.hit_rate / 100)
        print(
            f"[INFO] scenario={args.scenario} hit_rate={args.hit_rate}% prefill_input={prefill_input}",
            flush=True,
        )
        reset_prefix_cache(args.server_url)
        run_request(args, "prefill", prefill_input, 2)
        if args.prefill_sleep_s > 0:
            time.sleep(args.prefill_sleep_s)
    else:
        reset_prefix_cache(args.server_url)

    print("[INFO] Starting normal measurement phase", flush=True)
    summary = run_request(args, "normal", args.input_tokens, args.output_tokens)

    payload = {
        "scenario": args.scenario,
        "hit_rate": args.hit_rate,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "random_seed": args.random_seed,
        "server_url": args.server_url,
        "model": args.model,
        "tokenizer_path": args.tokenizer_path,
        "metrics": compact_metrics(summary),
        "summary": summary,
    }

    out_path = Path(args.results_dir) / f"{args.scenario}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[INFO] Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
