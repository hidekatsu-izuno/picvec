#!/usr/bin/env python3
"""Regenerate sample images and compare complete SVG bytes with a Git revision.

Run alone after building picvec; do not overlap conversions, builds, or tests.
Only exact, mask-free results may replace sample/output with --update-output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/picvec")
    parser.add_argument("--reference", default="HEAD")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--update-output", action="store_true")
    args = parser.parse_args()
    if args.threads < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("threads must be nonnegative and timeout positive and finite")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"executable not found: {binary}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.reference}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    available = {p.name: p for p in sorted((ROOT / "sample/input").iterdir())
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    names = args.inputs or list(available)
    if any(name not in available for name in names):
        parser.error("inputs must name PNG/JPEG files in sample/input")
    references = {name: subprocess.check_output(
        ["git", "show", f"{revision}:sample/output/{available[name].stem}.svg"], cwd=ROOT,
    ) for name in names}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    report = {"reference_commit": revision, "binary": str(binary),
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "requested_threads": args.threads, "runs": runs}
    for name in names:
        source = available[name]
        output = args.output_dir.resolve() / f"{source.stem}.svg"
        command = [str(binary), str(source), str(output),
                   "--threads", str(args.threads), "--verbose"]
        if source.stem == "cliparts-6x6":
            command.append("--remove-chroma-key-background")
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        with output.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                                         timeout=args.timeout)
                status = "ok" if process.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
        row = {"input": name, "seconds": time.perf_counter() - started,
               "status": status, "command": command}
        if status == "ok":
            data = output.read_bytes()
            tree = ET.fromstring(data)
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                       exact=data == references[name], mask_free=all(
                           e.tag.split("}")[-1] != "mask"
                           and all(k.split("}")[-1] != "mask" for k in e.attrib)
                           for e in tree.iter()))
            match = re.search(r'"execution_threads":\s*(\d+)',
                              output.with_suffix(".log").read_text())
            if match:
                row["execution_threads"] = int(match[1])
            if args.update_output and row["exact"] and row["mask_free"]:
                shutil.copyfile(output, ROOT / "sample/output" / output.name)
        runs.append(row)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if not all(r.get("exact") and r.get("mask_free") for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
