"""Convert sorted PNG frames in a directory into an mp4 video using OpenCV.

Usage:
  python scripts/make_video.py [--frames_dir DIR] [--out FILE] [--fps N]
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", default="results/pih_render",
                   help="directory containing frame_*.png files")
    p.add_argument("--out",        default="results/pih_render/episode.mp4")
    p.add_argument("--fps",        type=int, default=10,
                   help="output fps (default 10 — each frame is 0.1s sim at every=5)")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        sys.exit(f"No frame_*.png files found in {frames_dir}")

    sample = cv2.imread(str(paths[0]))
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        out.write(frame)

    out.release()
    print(f"Wrote {len(paths)} frames to {args.out}  ({args.fps} fps, {len(paths)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
