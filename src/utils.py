import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import argparse
import configparser
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax, SGD

import pickle
import os
import time
import numpy as np
from sklearn import metrics
import pandas as pd
import datetime
import io
from tqdm import tqdm
from sklearn.metrics import classification_report

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from time import perf_counter_ns
    print("[utils]:: perf_counter_ns exists. That means python >= 3.7")
except ImportError:
    print("[utils]:: perf_counter_ns does not exist. That means python < 3.7. I define perf_counter_ns based on datetime")
    def perf_counter_ns():
        now = datetime.datetime.now()
        return int(now.timestamp() * 1e9)

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0].parents[0])  # object_detection root directory
HOME = os.path.expanduser( '~' )
MODELS_PATH = ROOT + "/models/"
PLOTS_PATH = ROOT + "/report/img/"
HISTORY_PATH = ROOT + "/history/"

def measure_avg_exec_time(class_name):
    def time_decorator(func):
        nr_ignored_calls = 2
        num_iterations = 200 + nr_ignored_calls

        def wrapper(*args, **kwargs):
            # check if wrapper has the call_count attribute
            if not hasattr(wrapper, 'call_count'):
                # initialized counter
                wrapper.call_count = 0
                wrapper.total_time = 0

            if wrapper.call_count < num_iterations:

                t0 = perf_counter_ns()
                result = func(*args, **kwargs)
                t1 = perf_counter_ns()
                elapsed_time = t1-t0

                # ignore first call(s) cause they're usually garbage
                if wrapper.call_count >= nr_ignored_calls:
                    wrapper.total_time += elapsed_time
                
                wrapper.call_count += 1

                if wrapper.call_count == num_iterations:
                    average_duration_ms = wrapper.total_time / ( num_iterations - nr_ignored_calls ) * 1e-6
                    print(f"[utils]:: Time-stats: Avg exec time of method '{class_name}.{func.__name__}' over {num_iterations - nr_ignored_calls} calls: {average_duration_ms:.2f} [ms]")  
            else:
                # If the maximum number of calls has been reached,
                # it simply calls the original function without
                # performing the timing calculations.
                result = func(*args, **kwargs)
            return result
        return wrapper
    return time_decorator

def save_learning_curves(path, model_name, history, train):

    """Save learning curves of each teacher in a png.

    Args:
        path (str): Path where the plots associated with the
                    learning curves will be stored
        model_name (str): Unique name of the teacher
        history (dict): dictionary containing the training/validation
                        loss and accuracy after fitting the model
        train: boolean declaring whether the history variable comes
                directly from training or we load it from some npy file
                in a post-training fashion.
    """

    if train:
        history = history.history

    # Define needed variables
    tr_acc = history['acc']
    tr_loss = history['loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout

    # save plot to file
    if not os.path.exists(path):
        print("[utils]:: results will be stored under: " + path)
        os.makedirs(path)
    plt.savefig(path + model_name + '.png')
    plt.close()

def parse_opt():
    parser = argparse.ArgumentParser(description = "MRI variables")
    parser.add_argument("--config", default ='/config.ini', help="Path to the configuration file")
    config = read_config(ROOT + parser.parse_args().config)

    if config is None: 
        return None

    parser.add_argument("--mri_data_path", type = str, default = config['mri_data_path'], help = '')
    parser.add_argument('--augmentation', default = config['augmentation'], action = 'store_true', help = 'if set True augmented data will be used for training')
    parser.add_argument("--num_classes", type = str, default = config['num_classes'], help = '')
    parser.add_argument("--num_unfrozen_layers", type = str, default = config['num_unfrozen_layers'], help = '')

    parser.add_argument("--train_batch_size", type = int, default = config['train_batch_size'], help = 'specify size of batches to split the training data {16, 32, 64, 128...}')
    parser.add_argument('--test_size', type= float, default = config['test_size'], help = 'specify size of validation data')
    parser.add_argument("--epochs", type = int, default = config['epochs'], help = 'specify number of epochs')
    parser.add_argument("--learning_rate", type = float, default = config['learning_rate'], help = 'specify learning rate for the gradient descent algorithm')
    parser.add_argument("--input_size", type = int, default = config['input_size'], help = 'specify CNN input size : {128, 256, 512}')
    parser.add_argument("--model_name", type = str, default = config['model_name'], help = '')
    parser.add_argument("--model_type", type = str, default = config['model_type'], help = '')
    
    parser.add_argument("--test_batch_size", type = int, default = config['test_batch_size'], help = 'specify size of batches to split the testing data {16, 32, 64, 128...}')
    parser.add_argument("--test_sample", type = str, default = config['test_sample'], help = '')
    parser.add_argument("--test_label", type = str, default = config['test_label'], help = '')
    args = parser.parse_args()
    return args

# Reads from the config file and returns in a dictionary all the input variables
def read_config(config_file):
    
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        print(f"The file '{config_file}' does not exist.")
        print("The Default config file is train_cfg.ini ")
        config_file = ROOT + "/config/video_default.ini"

    config.read(config_file)  
    
    # Define a function to retrieve values from the specified section
    def get_config(section, key):
        try:
            return config[section][key]
        except KeyError:
            return None  # Return None if the key is not found       
    
    config_values = {
        # General Section
        'mri_data_path': get_config('General', 'mri_data_path'),
        'augmentation':  config.getboolean('General', 'augmentation'),
        'num_classes': get_config('General', 'num_classes'),

        # Training Section
        'train_batch_size': config.getint('Training', 'train_batch_size'),
        'test_size': config.getfloat('Training', 'test_size'),
        'epochs': config.getint('Training', 'epochs'),
        'learning_rate': config.getfloat('Training', 'learning_rate'),
        'input_size': config.getint('Training', 'input_size'),
        'num_unfrozen_layers': config.getint('Training', 'num_unfrozen_layers'),
        'model_name': get_config('Training', 'model_name'),
        'model_type': get_config('Training', 'model_type'),

        # Testing Section
        'test_batch_size': config.getint('Testing', 'test_batch_size'),
        'test_sample': get_config('Testing', 'test_sample'),
        'test_label': get_config('Testing', 'test_label'),
    }

    return config_values

def read_mri_data_train(args):
    """
    Reads the train data. We return the paths to load them
    later with Generators cause the memory can't handle such big data
    """

    if args.augmentation:
        print("[train]:: Training with augmented dataset.")
        train_dir = args.mri_data_path + '/train_augmented/'
    else:
        print("[train]:: Training with normal dataset.")
        train_dir = args.mri_data_path + '/train/'
    
    train_paths = []
    train_labels = []

    for label in os.listdir(train_dir):
        for image in os.listdir(train_dir+label):
            train_paths.append(train_dir+label+'/'+image)
            train_labels.append(label)

    train_paths, train_labels = shuffle(train_paths, train_labels)

    return (train_paths, train_labels)

def read_mri_data_test(args):
    """
    Reads the test data. We return the paths to load them
    later with Generators cause the memory can't handle such big data
    """

    if args.augmentation:
        print("[test]:: Test with augmented dataset.")
        test_dir = args.mri_data_path + '/test_augmented/'
    else:
        print("[test]:: Test with normal dataset.")
        test_dir = args.mri_data_path + '/test/'

    test_paths = []
    test_labels = []

    for label in os.listdir(test_dir):
        for image in os.listdir(test_dir+label):
            test_paths.append(test_dir+label+'/'+image)
            test_labels.append(label)

    test_paths, test_labels = shuffle(test_paths, test_labels)

    return (test_paths, test_labels)

def plot_confusion_matrix(model_name, labels_true, labels_pred):

    """Plot the confusion matrix. It shows how well the trained teacher
    predicts each class.

    Args:
        model_name (str): Name of the teacher
        labels_true (np.ndarray): Numpy array with true labels
        labels_pred (np.ndarray): Numpy array with predicted labels
    """

    classes_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    conf_matx = metrics.confusion_matrix(labels_true, labels_pred)
    conf_matx_edit = metrics.confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(10,10))
    ax = sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.xlabel('Predicted', fontsize = 16)
    plt.ylabel('Actual', fontsize = 16)

    diag = np.diag_indices(len(classes_names)) # indices of diagonal elements
    diag_sum = sum(conf_matx[diag]) # sum of diagonal elements
    off_diag_sum = np.sum(conf_matx) - diag_sum # sum of off-diagonal elements

    # row associated with the minimum diagonal element (this which class is problematic)
    row_of_min_diag_elem = np.argmin(conf_matx.diagonal())
    conf_matx_edit[row_of_min_diag_elem, row_of_min_diag_elem] = 0
    # column associated with the problematic class (this shows the erroneous prediction)
    col_with_max_off_diag_elem = np.argmax(conf_matx_edit[row_of_min_diag_elem, :]) 

    ax.set_xticklabels(classes_names, rotation=45)
    ax.set_yticklabels(classes_names, rotation=45)
    plt.title('model_name:{}'.format(model_name) + 
    '\n This model tends to erroneously predict [{}] instead of [{}]'.format(classes_names[col_with_max_off_diag_elem], classes_names[row_of_min_diag_elem]) +
     '\n {} out of {} predictions were erroneous (accuracy: {}%)'.format(off_diag_sum, labels_true.shape[0], round((100 - 100*off_diag_sum/labels_true.shape[0]),3)))
    # save plot to file
    plt.savefig("report/img/" + model_name + '_conf_mat.png')
    plt.close()

def plot_data_distribution(train_labels):
    plt.figure(figsize=(14,6))
    colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
    plt.rcParams.update({'font.size': 14})
    plt.pie([len([x for x in train_labels if x=='pituitary']),
            len([x for x in train_labels if x=='notumor']),
            len([x for x in train_labels if x=='meningioma']),
            len([x for x in train_labels if x=='glioma'])],
            labels=['pituitary','notumor', 'meningioma', 'glioma'],
            colors=colors, autopct='%.1f%%', explode=(0.025,0.025,0.025,0.025),
            startangle=30)
    # plt.show()
    plt.savefig("report/img/train_data_distribution.png")
    plt.close()

def plot_train_valid_distribution(args, train_labels, valid_labels):

    # Assuming label_encoder is the instance of LabelEncoder used for encoding
    # label_names contains the names of the classes in the same order as label_encoder.classes_
    label_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Reverse categorical labels to numerical form
    train_numerical_labels = np.argmax(train_labels, axis=1)
    valid_numerical_labels = np.argmax(valid_labels, axis=1)

    # Count occurrences of each numerical label in training and validation sets
    train_label_distribution = np.bincount(train_numerical_labels)
    valid_label_distribution = np.bincount(valid_numerical_labels)

    # Plotting the distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training set distribution
    ax1.bar(label_names, train_label_distribution, color='blue')
    ax1.set_title('Training Set Label Distribution')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Frequency')
    ax1.set_xticklabels(label_names, rotation=45)  # Rotate labels for better visibility

    # Validation set distribution
    ax2.bar(label_names, valid_label_distribution, color='orange')
    ax2.set_title('Validation Set Label Distribution')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Frequency')
    ax2.set_xticklabels(label_names, rotation=45)  # Rotate labels for better visibility

    plt.tight_layout()
    plt.savefig(f"report/img/{args.model_name}_train_valid_distribution.png")
    plt.close()

def plot_train_test_balance(train_labels, test_labels):
    plt.figure(figsize=(14,6))
    colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
    plt.rcParams.update({'font.size': 14})
    plt.pie([len(train_labels), len(test_labels)],
            labels=['Train','Test'],
            colors=colors, autopct='%.1f%%', explode=(0.05,0),
            startangle=30)
    # plt.show()
    plt.savefig("report/img/train_test_balance.png")
    plt.close()

def plot_random_images(X_train, Y_train):
    """
    Functions that plots 8 random images from the MRI dataset
    just to get a basic grasp of what we are dealing with
    """
    images = []
    for path in X_train[50:59]:
        image = load_img(path, target_size=(512, 512))
        images.append(image)
    
    labels = Y_train[50:59]
    fig = plt.figure(figsize=(12, 6))
    for x in range(1, 9):
        fig.add_subplot(2, 4, x)
        plt.axis('off')
        plt.title(labels[x])
        plt.imshow(images[x])
    plt.rcParams.update({'font.size': 12})
    plt.savefig("report/img/random_images.png")

def plot_augmented_images(X_train):
    """
    Function that saves a plot with the original and the applied
    augmentation to help the user understand how I have increased
    the size of the MRI dataset taken from Kaggle

    Args: 
        X_train: the absolute paths of images for some particular class (e.g., glioma)
    """
    images = []
    labels = []
    for path in X_train:
        if "Tr-gl_0011" in path:
            image = load_img(path, target_size=(256, 256))
            images.append(image)
            
            # Split the file path by '/'
            parts = path.split('/')

            # Get the filename
            filename = parts[-1]

            # Split the filename by '_'
            filename_parts = filename.split('_')

            # get technique
            technique = filename_parts[2].split('.')[0]
            labels.append("glioma_"+technique)
        
    fig = plt.figure(figsize=(10, 6))
    
    for x in range(1, 7):
        # Check if the label contains the word "original"
        if 'original' in labels[x-1]:
            ax = fig.add_subplot(2, 4, x)
            plt.axis('off')
            plt.title(labels[x-1])
            plt.imshow(images[x-1])
            
            # Draw a green box around the image
            rect = plt.Rectangle((0, 0), 256, 256, edgecolor='red', linewidth=10, fill=False)
            ax.add_patch(rect)
        else:
            fig.add_subplot(2, 4, x)
            plt.axis('off')
            plt.title(labels[x-1])
            plt.imshow(images[x-1])

    plt.rcParams.update({'font.size': 12})
    plt.savefig("report/img/aumgented_images.png")
    plt.close()

def plot_augmented_images_with_histogram(X_train):
    """
    Function that saves a plot with the original and the applied
    augmentation to help the user understand how I have increased
    the size of the MRI dataset taken from Kaggle

    Args: 
        X_train: the absolute paths of images for some particular class (e.g., glioma)
    """

    def bring_to_front(images_list, labels_list, specific_label):
        # Combine images and labels into tuples
        combined = list(zip(images_list, labels_list))

        # Sort the combined list based on labels
        combined.sort(key=lambda x: specific_label not in x[1])

        # Separate the sorted lists
        sorted_images, sorted_labels = zip(*combined)
        return list(sorted_images), list(sorted_labels)

    images = []
    labels = []
    for path in X_train:
        if "Tr-gl_0011" in path:

            # Split the file path by '/'
            parts = path.split('/')

            # Get the filename
            filename = parts[-1]

            # Split the filename by '_'
            filename_parts = filename.split('_')

            # get technique
            technique = filename_parts[2].split('.')[0]
            if technique != "mirror" and technique != "updown":
                labels.append("glioma_"+technique)

                image = img_to_array(load_img(path, target_size=(256, 256)))
                images.append(image)
            
    # Bring the original image and corresponding label to the front
    images, labels = bring_to_front(images, labels, "original")

    fig, axs = plt.subplots(4, 2, figsize=(14, 8))

    # draw a red rectangle around the  original image to distringuish it
    rect = plt.Rectangle((0, 0), 256, 256, edgecolor='red', linewidth=10, fill=False)
    axs[0, 0].add_patch(rect)

    # Loop through augmentations and plot images/histograms
    for i, (image, label) in enumerate(zip(images, labels)):
        
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)
        axs[i, 0].imshow(image)
        # axs[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(label)
        axs[i, 0].axis('off')

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        # Define the range for dark pixels (adjust as per your image)
        dark_pixel_range = (0, 50)  # Change this range to identify the dark region

        # Calculate the number of pixels falling in the dark region using the histogram
        dark_pixels_count = np.sum(hist[dark_pixel_range[0]:dark_pixel_range[1]])

        # Calculate the total number of pixels in the image
        total_pixels = image.shape[0] * image.shape[1]

        # Calculate the percentage of dark pixels
        percentage_dark_pixels = (dark_pixels_count / total_pixels) * 100

        # Find the x-position to place the percentage text
        text_x_pos = dark_pixel_range[1] - (dark_pixel_range[1] - dark_pixel_range[0]) * 0.8
        # Find the y-position to place the percentage text
        text_y_pos = max(hist) * 0.7

        # Display the percentage of dark pixels within the dark region
        axs[i, 1].text(text_x_pos, text_y_pos, f"{percentage_dark_pixels:.2f}%", color='black', fontsize=10)

        # Highlighting the dark region (adjust as per your image)
        axs[i, 1].axvspan(0, 50, color='black', alpha=0.15)  # Adjust the range to highlight darker pixels
        axs[i, 1].plot(hist, color='blue')
        # axs[i, 1].set_title(label)
        axs[i, 1].set_ylabel('Frequency')
        axs[i, 1].grid(True)
    axs[i, 1].set_xlabel('Gray Level')

    plt.rcParams.update({'font.size': 12})
    plt.savefig("report/img/aumgented_images_with_hist.png")
    plt.close()

def barbplot_distribution_train_val_test(Y_train, Y_valid, Y_test):
    
    # Example data (replace this with your actual data)

    # Finding unique values and their counts for each dataset
    _, train_counts = np.unique(Y_train, return_counts=True)
    _, valid_counts = np.unique(Y_valid, return_counts=True)
    _, test_counts = np.unique(Y_test, return_counts=True)

    # set width of bar 
    barWidth = 0.1
    fig = plt.subplots(figsize =(8, 6)) 
    
    # set height of bar 
    glioma = [train_counts[0], valid_counts[0], test_counts[0]] 
    meningioma = [train_counts[1], valid_counts[1], test_counts[1]] 
    notumor = [train_counts[2], valid_counts[2], test_counts[2]] 
    pituitary = [train_counts[3], valid_counts[3], test_counts[3]] 
    
    # Set position of bar on X axis 
    br1 = np.arange(len(glioma)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    
    # Make the plot
    plt.bar(br1, pituitary, color ='#4285f4', width = barWidth, 
        edgecolor ='grey', label ='pituitary') 
    plt.bar(br2, meningioma, color ='#fbbc05', width = barWidth, 
            edgecolor ='grey', label ='meningioma') 
    plt.bar(br3, glioma, color ='#34a853', width = barWidth, 
            edgecolor ='grey', label ='glioma') 
    plt.bar(br4, notumor, color ='#ea4335', width = barWidth, 
        edgecolor ='grey', label ='notumor') 
    
    # Adding Xticks 
    plt.title("Distribution of four conditions amongst train, test, and valid sets", fontweight ='bold', fontsize = 12)
    # plt.xlabel('', fontweight ='bold', fontsize = 12) 
    plt.ylabel('Counts', fontsize = 12) 
    plt.xticks([r + barWidth for r in range(len(glioma))], 
            ['Train', 'Valid', 'Test'], fontsize = 12)
    
    plt.legend()
    # plt.show() 
    plt.savefig("report/img/distribution_train_valid_test.png")
    plt.close()

# Function to load and preprocess images from file paths
def preprocess_images(file_paths, target_size):
    """
    This function loads and pre-process images. It basically resizes
    them and normalizes their pixel values between 0-1 to ensure numerical
    stability, faster convergence, better generalization and reduce
    computational load.

    Args:
        file_paths: absolute path of all images
        target_size: tuple in the form (int, int, int), e.g., (256, 256, 3)

    Returns:
        numpy array of images

    """
    images = []
    for file_path in file_paths:
        # Load image using OpenCV (you can also use PIL)
        image = cv2.imread(file_path)
        if image is not None:
            # Resize the image to the target size
            image = cv2.resize(image, target_size)
            # Perform any other preprocessing steps here (e.g., normalization)
            image = image / 255.0  # Normalize pixel values
            images.append(image)
    return np.array(images)

def preprocess_single_image(image_path, target_size):
    # Load image using OpenCV (you can also use PIL)
    image = cv2.imread(image_path)
    if image is not None:
        # Resize the image to the target size
        image = cv2.resize(image, target_size)
        # Perform any other preprocessing steps here (e.g., normalization)
        image = image / 255.0  # Normalize pixel values
        return image
    return None

def data_generator(file_paths, labels, batch_size, epochs, args):
    """
    Function to generate batches of images and labels
    This implementation handles the situation where the
    last batch might have fewer samples than the specified batch size.
    """
    num_samples = len(file_paths)
    num_batches = num_samples // batch_size

    for _ in range(epochs):

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            batch_paths = file_paths[start_idx:end_idx]
            batch_images = preprocess_images(batch_paths, (args.input_size, args.input_size))
            labels_batch = labels[start_idx:end_idx]

            yield batch_images, labels_batch

        # Handling the last batch
        if num_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            batch_paths = file_paths[start_idx:]
            batch_images = preprocess_images(batch_paths, (args.input_size, args.input_size))
            labels_batch = labels[start_idx:]

            yield batch_images, labels_batch

def get_compiled_model(args):
    """
    Function that based on the name of the model provided in the config file
    ruturns a the associated compiled model based on some research I have conducted.
    Feel free to modify the models or fine-tune their hyperparameters as you wish.
    """

    input_shape = (args.input_size, args.input_size, 3)

    # Remember, when fine-tuning a pre-trained model, especially with a small dataset,
    # it's generally recommended to only unfreeze and retrain a few of the final layers to prevent overfitting.
    # The more dataset you obtain (e.g., via data augmentation) the more layers you can unfreeze

    tf.keras.applications.xception.Xception

    if args.model_type == "Xception":
        base_model =   tf.keras.applications.xception.Xception(input_shape=input_shape, include_top=False, weights='imagenet')

        # last layers to keep unfrozen
        N = args.num_unfrozen_layers
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")
        
        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: https://www.kaggle.com/code/mushfirat/brain-tumor-classification-accuracy-96
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "Inceptionv3":
        base_model =   tf.keras.applications.inception_v3.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')

        # last layers to keep unfrozen
        N = args.num_unfrozen_layers
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")
        
        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: https://www.kaggle.com/code/mushfirat/brain-tumor-classification-accuracy-96
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "MobileNetv2":
        base_model =  tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

        # last layers to keep unfrozen
        N = args.num_unfrozen_layers
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")
        
        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: https://www.kaggle.com/code/mushfirat/brain-tumor-classification-accuracy-96
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif "EfficientNet" in args.model_type:

        # EfficientNet doesn't seem to be a good fit considering the limited dataset

        if args.model_type == "EfficientNetB0":
            base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
        elif args.model_type == "EfficientNetB1":
            base_model = tf.keras.applications.EfficientNetB1(input_shape=input_shape, include_top=False, weights='imagenet')
        elif args.model_type == "EfficientNetB7":
            base_model = tf.keras.applications.EfficientNetB7(input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            print(f"Currently supported only B0,B1,B7. {args.model_type} is not one of them")
            exit()

        # last layers to keep unfrozen
        N = args.num_unfrozen_layers
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")

        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: chatGPT
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "VGG16":
        base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

        # last layers to keep unfrozen (N=4 seem a good choice for vanilla train)
        N = args.num_unfrozen_layers
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")
        
        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: https://www.kaggle.com/code/mushfirat/brain-tumor-classification-accuracy-96
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "ResNet50":
        base_model = tf.keras.applications.resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        
        N = args.num_unfrozen_layers # 9 seems like a good choice for vanilla dataset
        total_layers = len(base_model.layers)
        print(f"Total number of {args.model_type} layers: {total_layers}")
        print(f"Total number of frozen {args.model_type} layers: {total_layers - N}")
        print(f"Total number of unfrozen {args.model_type} layers: {N}")
        
        # Freeze all layers 
        base_model.trainable = False
        if N > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-N:]:
                layer.trainable = True

        # inspired from: chatGPT
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(args.num_classes, activation='softmax'))

        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "large":
       # model inspired from: https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-/blob/master/CNN%20with%20Image%20Augmentation%20(CIFAR10).ipynb
        
        #reg=l2(1e-4)   # L2 or "ridge" regularisation
        reg = None
        num_filters = 32
        act='relu'
        opt = Adam(learning_rate = args.learning_rate)
        drop_dense = 0.5
        drop_conv = 0

        ## Initialize CNN model
        model = Sequential()

        ## Convolution Layers
        model.add(Conv2D(num_filters, (3, 3), activation=act, kernel_regularizer=reg, input_shape = input_shape, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(num_filters, (3, 3), activation=act,kernel_regularizer=reg, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
        model.add(Dropout(drop_conv))

        model.add(Conv2D(2*num_filters, (3, 3), activation=act, kernel_regularizer = reg, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(2*num_filters, (3, 3), activation=act, kernel_regularizer=reg, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 8x8x3x(2*num_filters)
        model.add(Dropout(drop_conv))

        model.add(Conv2D(4*num_filters, (3, 3), activation=act, kernel_regularizer = reg, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(4*num_filters, (3, 3), activation=act, kernel_regularizer = reg, padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
        model.add(Dropout(drop_conv))

        ## Converting from 2D --> 1D
        model.add(Flatten())

        ## Add Hidden Dense Layers
        model.add(Dense(512, activation=act, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Dropout(drop_dense))

        ## Output Dense Layer
        model.add(Dense(args.num_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    elif args.model_type == "tiny":
        # Initialize the CNN model
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten layer
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(args.num_classes, activation='softmax')) 

        # Compile model
        opt = Adam(learning_rate = args.learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
    else:
        print(f"Model {args.model_type} not supported")
        exit()

    return model

class Classifier:
    def __init__(self, args):
        self.args = args
        self.trained = False
        self.name = args.model_name
        self.model = None
        self.summary = None
        self.path = None
        self.history = None
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_colors = ['blueviolet', 'violet', 'green', 'orange']
        self.nb_classes = len(self.class_names)

    def compile(self):
        print(f"[utils][Classifier]:: Creating model with name {self.name}")

        self.model = get_compiled_model(self.args)

        self.summary = self.model.summary()

        return self.model
    
    def load(self, model_path, history_path):

        """Loading the saved model.

        Args:
            model_path (str): absolute path of the saved model
        """

        print(f"[Classifier]:: Loading {model_path}")
        self.path = model_path
        self.model = load_model(self.path)
        self.trained = True
        tmp_smry = io.StringIO()
        self.model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
        self.summary = tmp_smry.getvalue()
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_colors = ['blueviolet', 'violet', 'green', 'orange']
        self.nb_classes = len(self.class_names)
        self.history = np.load(history_path,allow_pickle='TRUE').item()

    def save(self, path, model_name):
        if not os.path.exists(path):
            print("[utils][Classifier]:: results will be stored under: " + path)
            os.makedirs(path)
    
        self.model.save(path + "/" + model_name)
        self.path = path + model_name

    def train(self, train_paths, train_labels):
        """
        Function that trains the model using the train data after splitting them in
        train and valid.

        Args:
            train_paths: paths of training MRI images
            train_labels: labels associated with each MRI image
        """

        print('='*70)
        print("[utils][Classifier]::Training of {} started ...".format(self.name))
        print('='*70)

        # If you find that the accuracy score remains at 10% after several epochs, 
        # try to re run the code. It’s probably because the initial random weights are just not good.

        if not os.path.exists(PLOTS_PATH):
            print("[utils][Classifier]:: history will be stored under: " + PLOTS_PATH)
            os.makedirs(PLOTS_PATH)

        # this is to avoid overfitting
        # if the validation accuracy does not increase for 10 consecutive steps the training will stop\
        # and will store the weights from the epoch that had the best performance
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

        # Convert labels to numerical form using LabelEncoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(train_labels)

        # Convert numerical labels to categorical form
        categorical_labels = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

        # Split the image paths and categorical labels into training and validation sets
        train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, categorical_labels, test_size=self.args.test_size, random_state=42, stratify = categorical_labels)

        print("===========================================================")
        print(f"Total amount of train images: {len(train_paths)}, train labels: {len(train_labels)}")
        print(f"Total amount of valid images: {len(valid_paths)}, valid labels: {len(valid_labels)}")

        plot_train_valid_distribution(self.args, train_labels, valid_labels)

        train_steps = len(train_paths)//self.args.train_batch_size
        valid_steps = len(valid_paths)//self.args.train_batch_size

        # Create generators for training and validation data to load the images
        # in batches for the RAM to handle the large volume
        train_generator = data_generator(train_paths, train_labels, self.args.train_batch_size, self.args.epochs, self.args)
        valid_generator = data_generator(valid_paths, valid_labels, self.args.train_batch_size, self.args.epochs, self.args)

        # Get a batch of data from the training generator
        batch_images, batch_labels = next(train_generator)

        # Display the shape of the batch
        print("Shape of train batch images:", batch_images.shape)
        print("Shape of train batch labels:", batch_labels.shape)

        # Get a batch of data from the validation generator
        batch_images, batch_labels = next(valid_generator)

        # Display the shape of the batch
        print("Shape of valid batch images:", batch_images.shape)
        print("Shape of valid batch labels:", batch_labels.shape)

        start_time = time.time()
        self.history = self.model.fit(train_generator,
                                steps_per_epoch=train_steps,
                                epochs=self.args.epochs, validation_data=valid_generator,
                                validation_steps=valid_steps,
                                verbose=2, callbacks=[es])
        
        np.save(HISTORY_PATH + self.args.model_name +".npy", self.history.history)

        print('='*40)
        print('[utils][Classifier]::Elapsed time during training: {} sec'.format(round(time.time() - start_time, 3)))
        print('='*40)
        self.trained = True
        return self.history

    # ground truth not necessary
    def make_single_prediction(self, image_path, true_label=None):
        """
        Function that makes a single prediction. It is used when we are utilizing
        the developed GUI.
        """

        # Preprocess the single image
        single_image = preprocess_single_image(image_path, (self.args.input_size, self.args.input_size))

        if single_image is not None:
            # Make a prediction on the preprocessed single image
            single_image = np.expand_dims(single_image, axis=0)  # Add batch dimension
            prediction = self.model.predict(single_image)

            # The prediction is in one-hot encoded format,
            # so need to covert it to numerical value
            predicted_class_index = np.argmax(prediction, axis=1)
            print(f"Processing image: {image_path}")
            if true_label is not None:
                print(f"true diagnosis: {true_label}, pred diagnosis: {self.class_names[predicted_class_index[0]]}")
            else:
                print(f"true diagnosis: unknown, pred diagnosis: {self.class_names[predicted_class_index[0]]}")
        else:
            print("Failed to load the image.")

        return self.class_names[predicted_class_index[0]]

    # ground truth essential
    def evaluate(self, test_paths, test_labels):

        """
        Function that tests the classifier against the test data and returns
        the classification report and a plot of the confusion matrix.
        """
    
        start_time = time.time()

        print('='*50)
        print("[utils][Classifier]::Evaluation of CNN started...")
        print('='*50)

        steps = len(test_labels) // self.args.test_batch_size

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Convert string labels to numerical form
        numerical_labels = label_encoder.fit_transform(test_labels)

        # convert labels into one-hot-encoded
        one_hot_labels = to_categorical(numerical_labels, num_classes=self.nb_classes)

        # Instantiate data generator with the one-hot-encoded labels
        test_generator = data_generator(test_paths, one_hot_labels, self.args.test_batch_size, 1, self.args)

        # Evaluate the model on test data
        num_batches_to_evaluate = len(test_paths)
        print(f"Evaluating for {steps * self.args.test_batch_size} samples")
        evaluation = self.model.evaluate(test_generator, steps=steps)

        # Print the evaluation metrics (e.g., loss and accuracy)
        print("Test Loss:", evaluation[0])
        print("Test Accuracy:", evaluation[1])

        test_generator = data_generator(test_paths, one_hot_labels, self.args.test_batch_size, 1, self.args)

        # Make predictions using the test generator
        print(f"Predicting for {steps * self.args.test_batch_size} samples")
        predictions = self.model.predict(test_generator, steps=steps)

        # Convert one-hot encoded predictions to numerical labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Extract true labels from the test generator
        true_labels = []

        test_generator = data_generator(test_paths, one_hot_labels, self.args.test_batch_size, 1, self.args)

        for i in range(steps):
            _, label_batch = next(test_generator)
            true_labels.extend(np.argmax(label_batch, axis=1))

        # Mapping from encoded labels to categorical labels
        label_mapping = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'} 

        decoded_true_labels = [label_mapping[label] for label in true_labels]
        decoded_predicted_labels = [label_mapping[label] for label in predicted_labels]

        # Generate classification report
        report = classification_report(decoded_true_labels, decoded_predicted_labels)

        plot_confusion_matrix(self.args.model_name, decoded_true_labels, decoded_predicted_labels)

        print("Classification Report:")
        print(report)