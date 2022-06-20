"""
Python post-processing tool for ocean-atmosphere coupled simulations.
"""

import argparse
import os
import sys
import logging
from post_processing import PostProcessing


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base_dir", type=str, help="Path to the simulation base directory."
    )

    args = parser.parse_args()

    run_dir = os.path.join(args.base_dir, "run")
    save_dir = os.path.join(args.base_dir, "processed")

    process = PostProcessing(run_dir)
    process.run(save_dir)
