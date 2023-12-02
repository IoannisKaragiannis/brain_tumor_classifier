import utils
import sys
import time
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0].parents[0])  # object_detection root directory
HOME = os.path.expanduser( '~' )
MODELS_PATH = ROOT + "/models/"
PLOTS_LC = ROOT + "/report/img/" # learning curves
HISTORY_PATH = ROOT + "/history/"

def main():
    # Load input arguments
    args = utils.parse_opt()

    if args is None:
        sys.exit()

    # instantiate model
    model = utils.Classifier(args)
    model.load(MODELS_PATH + args.model_name+".h5", HISTORY_PATH + args.model_name + ".npy")

    model.make_single_prediction(args.test_sample, args.test_label)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")