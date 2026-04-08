"""Check training progress: parse log for loss/MSE/eval metrics."""

import os
import re
import sys


def check_progress(log_path="runs/obstacles_v1/training.log",
                   run_dir="runs/obstacles_v1"):
    """Parse training log and report metrics."""
    # Check for checkpoints
    checkpoints = sorted(
        f for f in os.listdir(run_dir)
        if f.startswith("dagger_round") and f.endswith(".pt")
    ) if os.path.isdir(run_dir) else []

    print(f"=== Obstacle DAgger Training Status ===")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {len(checkpoints)}")
    if checkpoints:
        print(f"  Latest: {checkpoints[-1]}")

    if not os.path.exists(log_path):
        print(f"No log file at {log_path}")
        return

    with open(log_path) as f:
        log = f.read()

    # Extract round summaries
    rounds = re.findall(
        r"ROUND (\d+)/\d+.*?margin=([\d.]+).*?archive=(\d+) frames",
        log,
    )
    if rounds:
        print(f"\nRounds completed: {len(rounds)}")
        for rnum, margin, frames in rounds[-3:]:
            print(f"  Round {rnum}: margin={margin}, archive={frames} frames")

    # Extract training loss
    losses = re.findall(r"epoch\s+(\d+)/\d+\s+loss=([\d.]+)", log)
    if losses:
        last_epoch, last_loss = losses[-1]
        print(f"\nLatest training: epoch {last_epoch}, loss={last_loss}")

    # Extract eval results
    evals = re.findall(r"margin=([\d.]+):\s+(\d+)/(\d+)\s+=\s+(\d+)%", log)
    if evals:
        print(f"\nLatest eval results:")
        for margin, lands, total, pct in evals[-5:]:
            print(f"  margin={margin}: {lands}/{total} = {pct}%")

    # Check for errors
    errors = [l for l in log.split("\n") if "ERROR" in l or "Traceback" in l]
    if errors:
        print(f"\nErrors found:")
        for e in errors[-3:]:
            print(f"  {e.strip()}")

    # Check if IMPROVED or not
    improvements = re.findall(r"(IMPROVED!|No improvement).*", log)
    if improvements:
        print(f"\nMargin progression:")
        for imp in improvements[-5:]:
            print(f"  {imp.strip()}")

    # Check if complete
    if "COMPLETE" in log:
        print(f"\n*** TRAINING COMPLETE ***")
        final = re.search(r"final margin=([\d.]+)", log)
        if final:
            print(f"Final margin: {final.group(1)}")

    # Tail of log
    lines = log.strip().split("\n")
    print(f"\nLast 5 log lines:")
    for line in lines[-5:]:
        print(f"  {line}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runs/obstacles_v1/training.log"
    run_dir = os.path.dirname(log_path) or "runs/obstacles_v1"
    check_progress(log_path, run_dir)
