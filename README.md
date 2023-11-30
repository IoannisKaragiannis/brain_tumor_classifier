# brain_tumor_classifier

### Setup your environment

```terminal
cd ~/git_repos
git clone https://github.com/IoannisKaragiannis/brain_tumor_classifier.git

cd brain_tumor_classifier
chmod +x setup_env.sh
chmod +x remove_env.sh

./setup_env.sh

source ~/python_venv/brain/bin/activate
```

### Data Exploration

Perform data exploration 

```terminal
(brain)$ python3 src/data_exploration.py
```

The results will be stored under the `report/img` directory.

Below you can see an example of some random training data:

<img src="/home/ioannis/git_repos/brain_tumor_classifier/report/img/random_images.png" alt="image" style="zoom:72%;" />

We can also observe that the training data are quite balanced 

<img src="/home/ioannis/git_repos/brain_tumor_classifier/report/img/train_data_distribution.png" alt="image" style="zoom:72%;" />

Below one can observe that the distribution of the four different classes, namely `glioma`, `meningioma`, `notumor`, and `pituitary` is balanced among the training, validation and test data.

<img src="/home/ioannis/git_repos/brain_tumor_classifier/report/img/distribution_train_valid_test.png" alt="image" style="zoom:72%;" />

### Data Augmentation

1. Download the MRI data from [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

2. Unzip the file and copy it in the root directory of the current repo under the name `brain_tumor_mri_dataset`

3. Rename the subdirectories into `train` and `test` accordingly.

4. The data have different dimensions. For convenience while performing data augmentation I also resize them to 512x512.

5. To perform data augmentation do the following. You can add more augmentation techniques but be cautious when it comes to MRI dataset. If the distribution of the augmented dataset diverges significantly then the model will learn erroneously.

   ```terminal
   # augment train data
   (brain)$ python3 src/augment_data.py --data train
   
   # augment test data
   (brain)$ python3 src/augment_data.py --data test
   ```
   
   It will take a while. Next to the `train` and `test` folders, a new directory called `train_augmented` or `test_augmented`will be created accordingly. Only the training data or their augmented version will be used throughout the training process while the test data will be left untouched for evaluation purposes. An example of the different augmentation techniques applied on one particular glioma sample:

<img src="/home/ioannis/git_repos/brain_tumor_classifier/report/img/aumgented_images.png" alt="image" style="zoom:72%;" />

7. 



### Test a model

```terminal
# test single image
(brain)$ python3 src/test_single_mri.py

# test single image with GUI
(brain)$ python3 src/diagnose_with_gui.py

# evaluate model on test-set
(brain)$ python3 src/test.py
```

# RESULTS

Below you will find the performance of 4 different models, with and without data augmentation.

## ResNet50

```ini
[General]
mri_data_path=brain_tumor_mri_dataset
augmentation=False
num_classes=4

[Training]
train_batch_size=32
test_size=0.2
epochs=50
learning_rate=0.0001
input_size=224
model_name=ResNet50
num_unfrozen_layers=9
# allowed values: {tiny, large, VGG16, ResNet50, EfficientNetB{0,1,7}, MobileNetv2, Inceptionv3, Xception
model_type=ResNet50

[Testing]
test_batch_size=16
```



<img src="report/img/ResNet50.png" alt="image" style="zoom:72%;" />

classification report

```terminal
Test Loss: 0.3312033712863922
Test Accuracy: 0.8996913433074951
Predicting for 1296 samples
81/81 [==============================] - 121s 1s/step
QSocketNotifier: Can only be used with threads started with QThread
Classification Report:
              precision    recall  f1-score   support

      glioma       0.93      0.73      0.82       296
  meningioma       0.76      0.88      0.82       302
     notumor       0.96      0.99      0.97       402
   pituitary       0.96      0.97      0.96       296

    accuracy                           0.90      1296
   macro avg       0.90      0.89      0.89      1296
weighted avg       0.91      0.90      0.90      1296
```

confusion matrix

<img src="report/img/ResNet50_conf_mat.png" alt="image" style="zoom:72%;" />

## ResNet50-aug

```ini
[General]
mri_data_path=brain_tumor_mri_dataset
augmentation=True
num_classes=4

[Training]
train_batch_size=64
test_size=0.15
epochs=50
learning_rate=0.0001
input_size=224
model_name=ResNet50_aug
num_unfrozen_layers=15
# allowed values: {tiny, large, VGG16, ResNet50, EfficientNetB{0,1,7}, MobileNetv2, Inceptionv3, Xception
model_type=ResNet50

[Testing]
test_batch_size=16
```



<img src="report/img/ResNet50_aug.png" alt="image" style="zoom:72%;" />

classification report

```terminal
Test Loss: 0.3191703259944916
Test Accuracy: 0.9158604741096497
Predicting for 7856 samples
491/491 [==============================] - 709s 1s/step
QSocketNotifier: Can only be used with threads started with QThread
Classification Report:
              precision    recall  f1-score   support

      glioma       0.90      0.84      0.87      1797
  meningioma       0.84      0.83      0.83      1836
     notumor       0.95      0.99      0.97      2428
   pituitary       0.97      0.98      0.97      1795

    accuracy                           0.92      7856
   macro avg       0.91      0.91      0.91      7856
weighted avg       0.91      0.92      0.92      7856
```

confusion matrix

<img src="report/img/ResNet50_aug_conf_mat.png" alt="image" style="zoom:72%;" />

## VGG16

```ini
[General]
mri_data_path=brain_tumor_mri_dataset
augmentation=False
num_classes=4

[Training]
train_batch_size=32
test_size=0.2
epochs=50
learning_rate=0.0001
input_size=224
model_name=VGG16
num_unfrozen_layers=4
# allowed values: {tiny, large, VGG16, ResNet50, EfficientNetB{0,1,7}, MobileNetv2, Inceptionv3, Xception
model_type=VGG16

[Testing]
test_batch_size=16
```

<img src="report/img/VGG16.png" alt="image" style="zoom:72%;" />

classification report

```terminal
Test Loss: 0.20448878407478333
Test Accuracy: 0.9506173133850098
Predicting for 1296 samples
81/81 [==============================] - 2s 24ms/step
Classification Report:
              precision    recall  f1-score   support

      glioma       0.93      0.92      0.93       296
  meningioma       0.91      0.88      0.89       304
     notumor       0.99      1.00      0.99       401
   pituitary       0.96      0.99      0.97       295

    accuracy                           0.95      1296
   macro avg       0.95      0.95      0.95      1296
weighted avg       0.95      0.95      0.95      1296
```

confusion matrix

<img src="report/img/VGG16_conf_mat.png" alt="image" style="zoom:72%;" />

## VGG16-aug

```ini
[General]
mri_data_path=brain_tumor_mri_dataset
augmentation=True
num_classes=4

[Training]
train_batch_size=64
test_size=0.15
epochs=50
learning_rate=0.0001
input_size=224
model_name=VGG16_aug
num_unfrozen_layers=8
# allowed values: {tiny, large, VGG16, ResNet50, EfficientNetB{0,1,7}, MobileNetv2, Inceptionv3, Xception
model_type=VGG16

[Testing]
test_batch_size=16
```

<img src="report/img/VGG16_aug.png" alt="image" style="zoom:72%;" />

classification report

```terminal
Test Loss: 0.118095263838768
Test Accuracy: 0.976578414440155
Predicting for 7856 samples
491/491 [==============================] - 10s 20ms/step
Classification Report:
              precision    recall  f1-score   support

      glioma       0.97      0.95      0.96      1797
  meningioma       0.96      0.96      0.96      1834
     notumor       1.00      1.00      1.00      2428
   pituitary       0.98      1.00      0.99      1797

    accuracy                           0.98      7856
   macro avg       0.98      0.97      0.97      7856
weighted avg       0.98      0.98      0.98      7856
```

confusion matrix

<img src="report/img/VGG16_aug_conf_mat.png" alt="image" style="zoom:72%;" />

### GUI

```terminal
(brain)$ python3 src/diagnose_with_gui.py
```


<video width="640" height="360" controls>
  <source src="report/videos/gui_demo.webm" type="video/webm">
  Your browser does not support the video tag.
</video>