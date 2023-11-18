import cv2
import random 
import numpy as np

def rotate_image(image, angle):
    mean_color = np.mean(image)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Fill in the empty space after rotation
    rotated_image[rotated_image == 0] = mean_color

    return rotated_image

def deresolution_image(image, scale):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define the size of the new image
    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width <= 0 or new_height <= 0:
        return image
    
    # Descale image 
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Upcale image back
    resized_image = cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_AREA)

    return resized_image

def get_random_augmentation():
    return random.choice([
        lambda img: rotate_image(img, random.randint(-5, 5)),
        lambda img: cv2.GaussianBlur(img, (3, 3), 0),
        lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        lambda img: deresolution_image(img, random.uniform(0.7, 0.9)),
    ])