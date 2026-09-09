#!/usr/bin/env python3
"""Sweep OKLab palette thresholds using an already-built diagnostics release binary.

Run without concurrent builds/tests for meaningful timings. Outputs are written
only to the requested directory. The sweep selects thresholds; repeat selected
settings in reverse order for timing comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/release/picvec'))
    parser.add_argument('--inputs', nargs='+', default=[
        'sample/input/boy_and_turtle.png', 'sample/input/car.png',
        'sample/input/viewport1.jpg'])
    parser.add_argument('--scales', nargs='+', type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary_sha256 = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    rows = []
    for source in map(Path, args.inputs):
        for repeat in range(args.repeats):
            variants = list(args.scales)
            if repeat % 2:
                variants.reverse()
            for scale in variants:
                tag = f'oklab-{scale:g}'
                output = args.output / f'{source.stem}-{tag}-{repeat}.svg'
                command = [str(args.binary.resolve()), str(source.resolve()), str(output),
                           '--threads', '4', '--quality-metrics', '--verbose']
                command += ['--oklab-palette-threshold-scale', str(scale)]
                start = time.monotonic()
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                elapsed = time.monotonic() - start
                output.with_suffix('.log').write_text(result.stderr)
                # Progress precedes a pretty-printed JSON object; the final
                # human-readable status follows it.
                offset = next(i for i in range(len(result.stderr))
                              if result.stderr[i] == '{'
                              and (i == 0 or result.stderr[i - 1] == '\n'))
                summary, _ = json.JSONDecoder().raw_decode(result.stderr[offset:])
                row = dict(input=str(source), variant=tag, repeat=repeat,
                           command=command, binary_sha256=binary_sha256,
                           input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           wall_seconds=elapsed,
                           bytes=output.stat().st_size,
                           sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           summary=summary)
                rows.append(row)
                (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(source.name, tag, f'{elapsed:.3f}s', row['bytes'], summary['quality'], flush=True)


if __name__ == '__main__':
    main()
