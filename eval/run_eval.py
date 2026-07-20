#!/usr/bin/env python3
"""Run evaluation cases against the live chat API.

Usage:
    python eval/run_eval.py [--base-url https://localhost]

This script sends each case from eval/cases.json to POST /api/chat and checks
whether the response meets the expected criteria. Results are printed as a
table and the script exits with code 1 if any check fails.

Requires: httpx (pip install httpx)
"""

import argparse
import json
import ssl
import sys
from pathlib import Path

import httpx


def load_cases(path: str = "eval/cases.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def check_case(case: dict, response: dict) -> dict[str, bool]:
    """Run the defined checks against the API response."""
    results = {}
    checks = case.get("checks", {})

    if "prediction_present" in checks:
        results["prediction_present"] = response.get("prediction") is not None

    if "model_version_mentioned" in checks:
        has_version = response.get("model_version") is not None
        reply_has_version = "version" in (response.get("reply", "")).lower()
        results["model_version_mentioned"] = has_version or reply_has_version

    if "limitations_mentioned" in checks:
        reply = (response.get("reply", "")).lower()
        results["limitations_mentioned"] = any(
            kw in reply for kw in ["limitation", "historical", "demonstration", "not a real", "caution", "bias"]
        )

    if "asks_followup" in checks:
        reply = (response.get("reply", "")).lower()
        results["asks_followup"] = "?" in response.get("reply", "")

    if "no_fabricated_prediction" in checks:
        # If no tool was called, there should be no prediction
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            results["no_fabricated_prediction"] = response.get("prediction") is None
        else:
            results["no_fabricated_prediction"] = True

    if "no_causal_claim" in checks:
        reply = (response.get("reply", "")).lower()
        causal_words = ["caused", "because of", "the reason", "due to the fact"]
        results["no_causal_claim"] = not any(w in reply for w in causal_words)

    if "explains_correlation" in checks:
        reply = (response.get("reply", "")).lower()
        correlation_words = ["associat", "correlat", "historical", "pattern", "reflect"]
        results["explains_correlation"] = any(w in reply for w in correlation_words)

    if "states_demonstration_only" in checks:
        reply = (response.get("reply", "")).lower()
        results["states_demonstration_only"] = any(
            kw in reply for kw in ["demonstration", "not a real", "historical", "should not be used"]
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation cases")
    parser.add_argument("--base-url", default="https://localhost", help="API base URL")
    parser.add_argument("--cases", default="eval/cases.json", help="Path to cases JSON")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    # Allow self-signed certs
    client = httpx.Client(base_url=args.base_url, verify=False, timeout=60.0)

    total_checks = 0
    passed_checks = 0
    failed_cases = []

    print(f"\nRunning {len(cases)} evaluation cases against {args.base_url}")
    print("=" * 70)

    for case in cases:
        case_id = case["id"]
        message = case["input"]

        print(f"\n[{case_id}] {message[:60]}...")

        try:
            resp = client.post("/api/chat", json={"message": message})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_cases.append(case_id)
            continue

        # Check tool call expectation
        expected_tool = case.get("expected_tool_call")
        if expected_tool is not None:
            actual_tool = len(data.get("tool_calls", [])) > 0
            status = "PASS" if actual_tool == expected_tool else "FAIL"
            print(f"  tool_call_expected={expected_tool}, actual={actual_tool} -> {status}")
            total_checks += 1
            if status == "PASS":
                passed_checks += 1
            else:
                failed_cases.append(case_id)

        # Run content checks
        results = check_case(case, data)
        for check_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check_name} -> {status}")
            total_checks += 1
            if passed:
                passed_checks += 1
            elif case_id not in failed_cases:
                failed_cases.append(case_id)

        # Show truncated reply
        reply = (data.get("reply", ""))[:120]
        print(f"  Reply: {reply}...")

    print("\n" + "=" * 70)
    print(f"Results: {passed_checks}/{total_checks} checks passed")
    if failed_cases:
        print(f"Failed cases: {', '.join(failed_cases)}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
