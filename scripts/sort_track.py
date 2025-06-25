"""
sort_track.py

Script to run batch SORT tracking on YOLO detections using src/sort_utils.py.

Usage: python scripts/sort_track.py --detections_dir <detections> --output_dir <tracks>
"""

import argparse
from src.sort_utils import run_sort_on_detections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    run_sort_on_detections(args.detections_dir, args.output_dir)

if __name__ == "__main__":
    main()
