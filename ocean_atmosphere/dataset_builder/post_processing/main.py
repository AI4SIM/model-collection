from datasetBuilder import DatasetBuilder

import sys, logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    run_dir = "/net/172.16.118.188/data/ocean_atmosphere/simulations/wmed/run"
    save_dir = "/net/172.16.118.188/data/ocean_atmosphere/simulations/wmed/processed"

    builder = DatasetBuilder(run_dir)
    builder.run(save_dir)
