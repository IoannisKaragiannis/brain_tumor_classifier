import numpy as np
import os
import random
import argparse
import time
import cv2
from tqdm import tqdm
import imutils

def add_gaussian_noise(img, mean=0, std_dev=25):
    """
    Adds Gaussian noise to the image.
    
    Parameters:
    img (numpy.ndarray): Input MRI image (2D or 3D).
    mean (float): Mean of the Gaussian noise.
    std_dev (float): Standard deviation of the Gaussian noise.
    
    Returns:
    numpy.ndarray: Image with added Gaussian noise.
    """
    noise = np.random.normal(mean, std_dev, img.shape)
    noisy_image = img + noise
    # Ensure pixel values are within the valid range (0-255 for uint8 images)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_speckle_noise(img, var=0.1):
    """
    Adds speckle noise to the image.
    
    Parameters:
    image (numpy.ndarray): Input MRI image (3D volume).
    var (float): Variance of the speckle noise.
    
    Returns:
    numpy.ndarray: Image with added speckle noise.
    """
    noise = np.random.randn(*img.shape) * var
    noisy = img + img * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def rotate_img(img, direction):
    """
    Function returns a rotated image in clockwise and 
    andticlockwise direction
    """
    if direction == "clock":
        min_angle = -35  # Minimum rotation angle in degrees
        max_angle = -10   # Maximum rotation angle in degrees
    else:
        min_angle = 10  # Minimum rotation angle in degrees
        max_angle = 35   # Maximum rotation angle in degrees
    # Generate a random rotation angle
    rotation_angle = random.uniform(min_angle, max_angle)
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def crop_img3(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img

# cropping was creating artifacts in the image
# so I decided to better skip it
def crop_img(img):
    return img

def contrast_img(img, min_contrast, max_contrast):
    """
    Funtion that returns an image with randomly modified
    contrast within the specified boundaries

    contrast=1 ==> untouched image

    Args:
        img: original image
        min_contrast: minmimum contrast boundary
        max_contrast: maximum contrast boundary
    """
    # Convert the image to grayscale (if it's not already)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Generate a random contrast factor
    contrast_factor = random.uniform(min_contrast, max_contrast)
    # Modify the contrast of the image using the randomly generated factor
    aug_gray_image = np.clip(contrast_factor * gray_image, 0, 255).astype(np.uint8)
    # Convert the adjusted grayscale image back to a 3-channel image by replicating the single channel
    result = cv2.merge([aug_gray_image] * 3)
    return result

def brightness_img(img, min_brightness, max_brightness):
    """
    Function that returns an image with modified brightness

    Args:
        min_brightness: minimum brightness shift (subtractive)
        max_brightness: maximum brightness shift (additive)
    """

    # Generate a random brightness shift
    brightness_shift = random.randint(min_brightness, max_brightness)

    # Convert to float32 for higher accuracy in calculations
    img = img.astype('float32')

    # Increase brightness by adding a constant value (50 for example)
    result = cv2.add(img, brightness_shift)  # Adjust the value as needed

    # Clip pixel values to the valid range (0-255)
    result = np.clip(result, 0, 255).astype('uint8')

    return result

def shift_img(img, sign, direction):
    height, width = img.shape[:2]
    random_shift_x = random.randint(15, 30)
    random_shift_y = random.randint(15, 30)
    if sign == 1 and direction == "x":
        shift_right = random_shift_x
        right_translation_matrix = np.float32([[1, 0, shift_right], [0, 1, 0]])
        result = cv2.warpAffine(img, right_translation_matrix, (width, height))
    if sign == -1 and direction == "x":
        shift_left = random_shift_x
        left_translation_matrix = np.float32([[1, 0, -shift_left], [0, 1, 0]])
        result = cv2.warpAffine(img, left_translation_matrix, (width, height))
    if sign == 1 and direction == "y":
        shift_up = random_shift_y
        up_translation_matrix = np.float32([[1, 0, 0], [0, 1, -shift_up]])
        result = cv2.warpAffine(img, up_translation_matrix, (width, height))
    if sign == -1 and direction == "y":
        shift_down = random_shift_y
        down_translation_matrix = np.float32([[1, 0, 0], [0, 1, shift_down]])
        result = cv2.warpAffine(img, down_translation_matrix, (width, height))
    return result

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/", help="Which set to augment, either train or test")
    parser.add_argument("--mri_data_path", default="brain_tumor_mri_dataset/", help="The absolute folder path of the Visdrone Dataset")
    parser.add_argument("--all",default=True, action="store_true", help="For all augmentation techniques")
    parser.add_argument("--updown",default=False, action="store_true", help="For the upside-down augmentation technique")
    parser.add_argument("--brightness_up",default=False, action="store_true", help="For the brightness decrease distortion augmentation technique")
    parser.add_argument("--brightness_down",default=False, action="store_true", help="For the brightness decrease distortion augmentation technique")
    parser.add_argument("--contrast_up",default= False, action="store_true", help="For contrast increase distortion technique")
    parser.add_argument("--contrast_down",default= False, action="store_true", help="For contrast increase distortion technique")
    parser.add_argument("--mirror",default=False, action="store_true", help="For the mirror augmentation technique")
    parser.add_argument("--rot_clock",default=False, action="store_true", help="For clockwise rotation augmentation technique")
    parser.add_argument("--rot_anticlock",default=False, action="store_true", help="For anti-clockwise rotation augmentation technique")
    parser.add_argument("--shift",default=False, action="store_true", help="For shifting image left/right/top/bottom by random pixels")
    parser.add_argument("--gaussian_noise",default=False, action="store_true", help="Add gaussian noise")
    parser.add_argument("--speckle_noise",default=False, action="store_true", help="Add speckle noise")
    args = parser.parse_args()

    if not os.path.exists(args.mri_data_path):
        print('You gave wrong Paths')
        return

    if args.data == "train":
        train_test = 'train'
    elif args.data == "test":
        train_test = "test"
    else:
        print(f"[augment_data]:: data can either be 'train' or 'test' but you gave {args.data}")
        exit()

    train_dir = args.mri_data_path + f'{train_test}/'
    
    aug_glioma_dir = args.mri_data_path + f'/{train_test}_augmented/glioma'
    aug_meningioma_dir =  args.mri_data_path + f'/{train_test}_augmented/meningioma'
    aug_notumor_dir = args.mri_data_path + f'/{train_test}_augmented/notumor'
    aug_pituitary_dir = args.mri_data_path + f'/{train_test}_augmented/pituitary'

    # Create augmentated training folder
    if not os.path.exists(args.mri_data_path + f'/{train_test}_augmented/'):
        os.makedirs(args.mri_data_path + f'/{train_test}_augmented/')
        os.makedirs(aug_glioma_dir)
        os.makedirs(aug_meningioma_dir)
        os.makedirs(aug_notumor_dir)
        os.makedirs(aug_pituitary_dir)

    # Define a target size for resizing the images and the desired number of channels (e.g., 3 for RGB)
    # For now I realized that most of them were having dimensions 512x512x3.
    # However, since I am about to increase the training dataset by 10 times, and considering that no CNN
    # can have such a great input, I decided to downscale all of them to reasonable smaller
    # resolution 256x256x3 to occupy less space in my drive without losing too much information.
    target_size = (256, 256)

    for label in tqdm(os.listdir(train_dir), desc="Labels"):
        for image in tqdm(os.listdir(train_dir + label), desc="Images"):

            # read image with opencv
            image_original = cv2.imread(train_dir + label + '/' + image)
            
            aug_images = []
            aug_techniques = []

            # copy original image
            aug_images.append(crop_img(image_original))
            aug_techniques.append("original")

            # apply augmentation techniques

            if args.mirror or args.all:
                aug_image = cv2.flip(crop_img(image_original), 1)
                aug_images.append(aug_image)
                aug_techniques.append("mirror")

            # if args.rot_clock or args.all:
            #     aug_image = rotate_img(image_original, "clock")
            #     aug_image = crop_img(aug_image)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("rot-clock")

            # if args.rot_anticlock or args.all:
            #     aug_image = rotate_img(image_original, "anticlock")
            #     aug_image = crop_img(aug_image)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("rot-anticlock")
            #     aug_image = contrast_img(aug_image, 1.2, 1.4)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("rot-anticlock-contr-up")

            if args.updown or args.all:
                aug_image = cv2.flip(image_original, 0)
                aug_image = crop_img(aug_image)
                aug_images.append(aug_image)
                aug_techniques.append("updown")

            # if args.shift or args.all:
            #     aug_image = shift_img(crop_img(image_original), -1, "x")
            #     aug_image = crop_img(aug_image)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("shift-left")

            #     aug_image = shift_img(crop_img(image_original), 1, "x")
            #     aug_images.append(aug_image)
            #     aug_techniques.append("shift-right")

            #     aug_image = shift_img(crop_img(image_original), 1, "y")
            #     aug_images.append(aug_image)
            #     aug_techniques.append("shift-up")

            #     aug_image = shift_img(crop_img(image_original), -1, "y")
            #     aug_images.append(aug_image)
            #     aug_techniques.append("shift-down")

            if args.contrast_up or args.all:
                aug_image = contrast_img(image_original, 1.2, 1.4)
                aug_images.append(aug_image)
                aug_techniques.append("contr-up")

            if args.contrast_down or args.all:
                aug_image = contrast_img(image_original, 0.6, 0.8)
                aug_images.append(aug_image)
                aug_techniques.append("contr-down")

            # if args.brightness_up or args.all:
            #     aug_image = brightness_img(image_original, 20, 50)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("bright-up")

            # if args.brightness_down or args.all:
            #     aug_image = brightness_img(image_original, -50, -20)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("bright-down")

            if args.gaussian_noise or args.all:
                aug_image = add_gaussian_noise(image_original, 0, 15)
                aug_images.append(aug_image)
                aug_techniques.append("gaussian")

            # if args.speckle_noise or args.all:
            #     aug_image = add_speckle_noise(image_original, 0.4)
            #     aug_images.append(aug_image)
            #     aug_techniques.append("speckle")

            for img, technique in zip(aug_images, aug_techniques):

                # Resize the image to a uniform size
                # By default images are of different size and this
                # may cause issues later on during training of the CNN
                img = cv2.resize(img, target_size)

                # print(img.shape)

                img_full_path = args.mri_data_path + f'/{train_test}_augmented/' + label + '/' + image.split('.')[0] + '_' + technique + '.jpg'
                cv2.imwrite(img_full_path, img)     
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")