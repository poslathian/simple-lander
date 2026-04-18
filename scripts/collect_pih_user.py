"""Phase 2.5 user-correction batch workflow (Component 4).

Reports collection progress and launches the interactive viewer for the
next uncorrected episode.  The viewer handles one episode at a time;
run this script repeatedly (or just keep the viewer open and press N/A).

Usage
-----
  # Show progress stats only
  python scripts/collect_pih_user.py --stats

  # Launch viewer starting at the first uncorrected episode
  python scripts/collect_pih_user.py

  # Launch viewer at a specific episode index
  python scripts/collect_pih_user.py --episode 5

Arguments
---------
  --failures  JSONL of failure episodes to correct (default: results/phase2_5/failures.jsonl)
  --out       JSONL to write corrections to       (default: results/phase2_5/user_corrections.jsonl)
  --episode   Force-start at this 0-indexed episode (default: first uncorrected)
  --stats     Print progress and exit without launching viewer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_seeds(path: str | Path) -> list[int]:
    seeds = []
    p = Path(path)
    if not p.exists():
        return seeds
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    seeds.append(json.loads(line)["seed"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seeds


def print_stats(failures_path: str, out_path: str) -> list[int]:
    failure_seeds = load_seeds(failures_path)
    corrected_seeds = set(load_seeds(out_path))

    n_total     = len(failure_seeds)
    n_corrected = sum(1 for s in failure_seeds if s in corrected_seeds)
    n_remaining = n_total - n_corrected

    # Success rate among corrected episodes
    n_success = 0
    out_p = Path(out_path)
    if out_p.exists():
        with open(out_p) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("termination_reason") == "success":
                            n_success += 1
                    except json.JSONDecodeError:
                        pass

    print(f"\nPhase 2.5 correction progress")
    print(f"  Failures:   {n_total}")
    print(f"  Corrected:  {n_corrected}  ({n_corrected/n_total:.0%})")
    print(f"  Remaining:  {n_remaining}")
    if n_corrected > 0:
        print(f"  Success rate of replays: {n_success}/{n_corrected} "
              f"({n_success/n_corrected:.0%})")
    print()

    return failure_seeds, corrected_seeds


def first_uncorrected_idx(failure_seeds: list[int], corrected_seeds: set[int]) -> int:
    for i, s in enumerate(failure_seeds):
        if s not in corrected_seeds:
            return i
    return 0  # all corrected — restart from 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", default="results/phase2_5/failures.jsonl")
    ap.add_argument("--out",      default="results/phase2_5/user_corrections.jsonl")
    ap.add_argument("--episode",  type=int, default=None,
                    help="Force-start at this episode index (default: first uncorrected)")
    ap.add_argument("--stats",    action="store_true",
                    help="Print progress stats and exit")
    args = ap.parse_args()

    failure_seeds, corrected_seeds = print_stats(args.failures, args.out)

    if args.stats:
        return

    if not Path(args.failures).exists():
        print(f"ERROR: failures file not found: {args.failures}")
        sys.exit(1)

    start_ep = args.episode
    if start_ep is None:
        start_ep = first_uncorrected_idx(failure_seeds, corrected_seeds)
        print(f"Starting at first uncorrected episode: index {start_ep} "
              f"(seed {failure_seeds[start_ep] if failure_seeds else '?'})\n")

    viewer = Path(__file__).parent / "interactive_viewer.py"
    cmd = [
        sys.executable, str(viewer),
        "--jsonl",   args.failures,
        "--episode", str(start_ep),
        "--out",     args.out,
    ]
    print(f"Launching viewer: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
