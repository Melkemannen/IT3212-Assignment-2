import cv2
import numpy as np
from skimage import color
from skimage.feature import hog
from skimage import data, exposure, io
import matplotlib.pyplot as plt

#1. Write a Python script to compute the HOG features of a given image using a library such as OpenCV or scikit-image.
def hog_features(image, resize_dim=(256, 256)):
    image_resized = cv2.resize(image, resize_dim)
    image_gray = color.rgb2gray(image_resized) # Converting image to grayscale
    features, hog_image = hog(image_gray, orientations=12, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    return image_resized, hog_image, image_gray



#2. Apply your implementation to at least three different images, including both simple and complex scenes.
image1 = cv2.imread('vehicle_types\hatchback\PIC_141.jpg') #simple
image2 = cv2.imread('vehicle_types\pickup\PIC_146.jpg') #kinda complex
image3 = cv2.imread('vehicle_types\motorcycle\PIC_41.jpg') #complex

image_resized1, hog_image1, image1_gray = hog_features(image1)
image_resized2, hog_image2, image2_gray = hog_features(image2)
image_resized3 ,hog_image3, image3_gray = hog_features(image3)

#3. Visualize the original image, the gradient image, and the HOG feature image of the three images above.
# Visualize the original image, the grayscale image, and the HOG feature image of the three images above.
plt.figure(figsize=(18, 12))

# Original images
plt.subplot(3, 3, 1)
plt.title('Original Image 1')
plt.imshow(cv2.cvtColor(image_resized1, cv2.COLOR_BGR2RGB))

plt.subplot(3, 3, 4)
plt.title('Original Image 2')
plt.imshow(cv2.cvtColor(image_resized2,cv2.COLOR_BGR2RGB))

plt.subplot(3, 3, 7)
plt.title('Original Image 3')
plt.imshow(cv2.cvtColor(image_resized3, cv2.COLOR_BGR2RGB))

# Grayscale images
plt.subplot(3, 3, 2)
plt.title('Grayscale Image 1')
plt.imshow(image1_gray, cmap='gray')

plt.subplot(3, 3, 5)
plt.title('Grayscale Image 2')
plt.imshow(image2_gray, cmap='gray')

plt.subplot(3, 3, 8)
plt.title('Grayscale Image 3')
plt.imshow(image3_gray, cmap='gray')

# HOG images
plt.subplot(3, 3, 3)
plt.title('HOG Image 1')
plt.imshow(hog_image1, cmap='gray')

plt.subplot(3, 3, 6)
plt.title('HOG Image 2')
plt.imshow(hog_image2, cmap='gray')

plt.subplot(3, 3, 9)
plt.title('HOG Image 3')
plt.imshow(hog_image3, cmap='gray')

plt.tight_layout()
plt.show()




#4. Compare the HOG features extracted from different images. Discuss the impact of varying parameters like cell size, block size, and the number of bins on the resulting HOG descriptors.