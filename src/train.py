# Miscellaneous
import os
from pathlib import Path
import sys
import time
import utils

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0].parents[0])  # object_detection root directory
HOME = os.path.expanduser( '~' )
MODELS_PATH = ROOT + "/models/"
PLOTS_PATH = ROOT + "/report/img/"
HISTORY_PATH = ROOT + "/history/"

def main():

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    if not os.path.exists(HISTORY_PATH):
        os.makedirs(HISTORY_PATH)

    # Load input arguments
    args = utils.parse_opt()

    if not os.path.exists(ROOT + "/config"):
        os.makedirs(ROOT + "/config")

    # open both files 
    with open(ROOT + '/config.ini','r') as firstfile, open(ROOT+f"/config/{args.model_name}.ini",'w') as secondfile: 
        # read content from first file 
        for line in firstfile: 
            # append content to second file (I do this to remember the configuration of each training)
            secondfile.write(line)

    if args is None:
        print(f"[train]:: Failed to parse config file!")
        sys.exit()

    # load images
    train_paths, train_labels = utils.read_mri_data_train(args)

    print("===========================================================")
    print(f"Total amount of train images: {len(train_paths)}")

    # barbplot_distribution_train_val_test(train_label_encoder.inverse_transform(np.argmax(train_labels, axis=1)), train_label_encoder.inverse_transform(np.argmax(valid_labels, axis=1)), test_labels)

    # instantiate model 
    model = utils.Classifier(args)
    # compile model
    model.compile()
    print(model.summary)
    model.train(train_paths, train_labels)
    utils.save_learning_curves(PLOTS_PATH, model.name, model.history, train = True)
    model.save(MODELS_PATH, args.model_name + ".h5")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")