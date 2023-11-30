import utils
import sys
import time
import os
from pathlib import Path
import random

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

    # Target Input Size
    target_cnn_shape = (args.input_size, args.input_size, 3)

    # instantiate model
    model = utils.Classifier(args)
    model.load(MODELS_PATH + args.model_name+".h5", HISTORY_PATH + args.model_name + ".npy")
   
    # evaluate your model against a single sample
    # Directory containing the test dataset {glioma, meningioma, notumor, pituitary}
    test_path = args.mri_data_path + '/glio4.jpg'
    # test_path = args.mri_data_path + "/test/glioma/Te-gl_0010.jpg"
    test_label = "glioma"

    model.make_single_prediction(test_path, test_label)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")