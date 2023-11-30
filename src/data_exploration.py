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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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
PLOTS_LC = ROOT + "/report/img/" # learning curves

def main():
    
    if not os.path.exists(PLOTS_LC):
        os.makedirs(PLOTS_LC)

    # Load input arguments
    args = utils.parse_opt()
    
    if args is None:
        sys.exit()

    # load images
    train_paths, train_labels = utils.read_mri_data_train(args)
    test_paths, test_labels = utils.read_mri_data_test(args)

    if args.augmentation:
        utils.plot_augmented_images(train_paths)

    utils.plot_data_distribution(train_labels)
    utils.plot_random_images(train_paths, train_labels)

    # Split the image paths and categorical labels into training and validation sets
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=args.test_size, random_state=42)

    print("="*70)
    print(f"Total amount of train images: {len(train_paths)}, train labels: {len(train_labels)}")
    print(f"Total amount of valid images: {len(valid_paths)}, valid labels: {len(valid_labels)}")
    print(f"Total amount of test images: {len(test_paths)} test labels: {len(test_labels)}")
    print("="*70)

    utils.barbplot_distribution_train_val_test(train_labels, valid_labels, test_labels)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")