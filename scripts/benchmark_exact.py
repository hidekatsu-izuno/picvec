#!/usr/bin/env python3
"""Compare two picvec executables without replacing committed sample outputs.

Run serially, alternating executable order. Every trial checks SVG bytes and
all diagnostic fields except elapsed time and the output pathname.
With --samples, discover every sample input and also check committed SVGs.
--max-regression-seconds rejects any individual pair reaching that slowdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "booster": ("sample/input/booster-layout.jpg", []),
    "cliparts": ("sample/input/cliparts.png", []),
    "still-life": ("sample/input/vectorization-stress-still-life.png", []),
    "car": ("sample/input/car.png", []),
    "boy": ("sample/input/boy_and_turtle.png", []),
    "photo": ("sample/input/viewport2.jpg", []),
    "photo-small": ("sample/input/viewport1.jpg", []),
    "wiki": ("sample/input/wikipedia_logo_1_0.png", []),
    "alpha": ("src/test-data/cube-alpha.png", []),
    "key": ("src/test-data/round-buttons-source.png", ["--remove-chroma-key-background"]),
    "adaptive": ("sample/input/cliparts-6x6.png", ["--remove-chroma-key-background"]),
    "adaptive-small": ("sample/input/cliparts.png", ["--max-dimension", "512"]),
    "car-3passes": ("sample/input/car.png", ["--paint-merge-passes", "3"]),
    "car-quality": ("sample/input/car.png", ["--quality-metrics"]),
}


def run(binary: Path, case: str, output: Path, threads: int,
        timeout: float | None = None) -> dict:
    source, options = CASES[case]
    command = [str(binary), str(ROOT / source), str(output),
               "--threads", str(threads), "--verbose", *options]
    started = time.perf_counter()
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    log = output.with_suffix(".log").read_text()
    summary, _ = json.JSONDecoder().raw_decode(log[log.index("{\n"):])
    summary.pop("elapsed_seconds")
    summary.pop("output")
    return {
        "seconds": elapsed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "summary": summary,
        "stages": {name: float(value) for name, value in
                   re.findall(r"picvec stage ([\w-]+): ([\d.]+)s", log)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory for SVGs, logs and results.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cases", nargs="+", choices=CASES)
    selection.add_argument("--samples", action="store_true",
                           help="Compare every sample/input image with default sample options")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float,
                        help="Optional time limit per conversion; a timeout fails the comparison")
    parser.add_argument("--max-regression-seconds", type=float,
                        help="Reject if any candidate trial is this many seconds slower or worse")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 0:
        parser.error("repeats must be positive and threads nonnegative")
    if args.max_regression_seconds is not None and (
        not math.isfinite(args.max_regression_seconds) or args.max_regression_seconds <= 0
    ):
        parser.error("maximum regression must be positive and finite")
    if args.timeout_seconds is not None and (
        not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
    ):
        parser.error("timeout must be positive and finite")
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f"executable not found: {binary}")
    committed = {}
    revision = None
    if args.samples:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        args.cases = []
        for source in sorted((ROOT / "sample/input").iterdir()):
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            case = f"sample-{source.name}"
            options = ["--remove-chroma-key-background"] if source.stem == "cliparts-6x6" else []
            CASES[case] = (str(source.relative_to(ROOT)), options)
            args.cases.append(case)
            data = subprocess.check_output(
                ["git", "show", f"{revision}:sample/output/{source.stem}.svg"], cwd=ROOT)
            committed[case] = hashlib.sha256(data).hexdigest()
        if not args.cases:
            parser.error("no sample images found")
    elif args.cases is None:
        args.cases = ["car", "wiki", "boy", "alpha"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    references = {}
    report = {
        "executables": {key: {"path": str(value),
                              "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
                        for key, value in binaries.items()},
        "threads": args.threads,
        "reference_commit": revision,
        "committed_svg_sha256": committed,
        "max_regression_seconds": args.max_regression_seconds,
        "regressions": [],
        "runs": records,
    }
    result_path = args.output_dir / "results.json"
    for repeat in range(args.repeats):
        for case in args.cases:
            order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            pair = {}
            for version in order:
                output = args.output_dir / f"{case}-{repeat}-{version}.svg"
                print(f"START {case} {repeat + 1} {version}", flush=True)
                trial = run(binaries[version], case, output, args.threads, args.timeout_seconds)
                records.append({"case": case, "repeat": repeat, "version": version, **trial})
                pair[version] = trial
                result_path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{case} {repeat + 1} {version}: {trial['seconds']:.3f}s", flush=True)
            baseline = pair["baseline"]
            references.setdefault(case, baseline)
            for trial in pair.values():
                reference = references[case]
                if (trial["sha256"] != reference["sha256"]
                    or trial["summary"] != reference["summary"]
                    or (case in committed and trial["sha256"] != committed[case])):
                    raise SystemExit(f"Output differs for {case}; inspect {result_path}")
            regression = pair["candidate"]["seconds"] - baseline["seconds"]
            if args.max_regression_seconds is not None and regression >= args.max_regression_seconds:
                report["regressions"].append({"case": case, "repeat": repeat,
                                              "seconds": regression})
                result_path.write_text(json.dumps(report, indent=2) + "\n")
                raise SystemExit(f"Rejected: {case} slowed by {regression:.3f}s; inspect {result_path}")
    report["all_outputs_equal"] = True
    report["medians"] = {}
    for case in args.cases:
        medians = {version: statistics.median(row["seconds"] for row in records
                    if row["case"] == case and row["version"] == version)
                   for version in binaries}
        report["medians"][case] = {
            **medians,
            "reduction_percent": 100 * (1 - medians["candidate"] / medians["baseline"]),
        }
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["medians"], indent=2), flush=True)


if __name__ == "__main__":
    main()
