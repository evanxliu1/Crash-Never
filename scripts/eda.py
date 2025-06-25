"""
eda.py

Script to perform EDA on the Nexar dataset using utilities from src/eda_utils.py.

Usage: python scripts/eda.py --csv <train.csv>
"""

import argparse
from src.eda_utils import analyze_class_distribution, plot_event_timing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()

    analyze_class_distribution(args.csv)
    plot_event_timing(args.csv)

if __name__ == "__main__":
    main()
