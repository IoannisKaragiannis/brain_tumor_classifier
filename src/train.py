# For Data Processing
import numpy as np

# For ML Models
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *

# Miscellaneous
import os
from pathlib import Path
import sys
import cv2
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
                # append content to second file 
                secondfile.write(line)

    if args is None:
        sys.exit()

    # Target Input Size
    target_cnn_shape = (args.input_size, args.input_size, 3)

    # load images
    train_paths, train_labels = utils.read_mri_data_train(args)

    # I have in purpose augmented and resized the images to 512x512x3
    original_image_shape = (-1, -1, -1)
    for image in train_paths:
        original_image_shape = cv2.imread(image).shape
        break
    print(f"[train]:: image_shape: {original_image_shape}")
    print(f"[train]:: target_cnn_shape: {target_cnn_shape}")

    # utils.plot_data_distribution(train_labels)
    # utils.plot_random_images(train_paths, train_labels)

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