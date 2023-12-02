import utils
import sys
import time
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0].parents[0])  # object_detection root directory
HOME = os.path.expanduser( '~' )
MODELS_PATH = ROOT + "/models/"
PLOTS_PATH = ROOT + "/report/img/"
HISTORY_PATH = ROOT + "/history/"

def main():

    # Load input arguments
    args = utils.parse_opt()
    
    if args is None:
        sys.exit()

    # load test images
    test_paths, test_labels = utils.read_mri_data_test(args)

    print("===========================================================")
    print(f"Total amount of test images: {len(test_paths)}")

    # instantiate model
    model = utils.Classifier(args)
    # model = utils.TransferLearning(target_cnn_shape, "ResNet50", args)
    model.load(MODELS_PATH + args.model_name+".h5", HISTORY_PATH + args.model_name + ".npy")
    # utils.save_learning_curves(HISTORY_PATH, model.name, model.history, train = False)

    # evaluate your model against all test samples
    model.evaluate(test_paths, test_labels)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")