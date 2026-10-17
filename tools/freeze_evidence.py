"""Explicitly pin current local artifacts; review changes before accepting a new baseline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMBINED = "datasets/math/test_l2_l3_remaining_v1/combined_analysis_v32_final"
MODELS = ("base", "sft_v3", "grpo1024", "grpo1280", "grpo1536")


def source(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def main():
    target = ROOT / "configs/evidence_registry.json"
    if target.exists():
        raise SystemExit("Registry exists. Preserve it; explicitly move it before re-freezing inputs.")
    datasets = {}
    for dataset, folder, count in (("math", "math_v32", 2024), ("gsm8k", "gsm8k_v31", 1319)):
        datasets[dataset] = {"expected_rows": count,
            "models": {model: source(f"{COMBINED}/{folder}/{model}.jsonl") for model in MODELS}}
    datasets["math"]["models"]["sft_v4"] = source("datasets/math/sft_v4_vs_base_v32/results/sft_v4_scored_v32.jsonl")
    datasets["math"]["remaining_ids"] = source("datasets/math/sft_v4_vs_base_v32/questions_remaining.jsonl")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({"schema": "r1-evidence-registry-v1", "datasets": datasets}, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
