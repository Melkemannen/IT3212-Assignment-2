import cv2
import numpy as np
from skimage import color
from skimage.feature import hog
from skimage import data, exposure, io
import matplotlib.pyplot as plt

#1. Write a Python script to compute the HOG features of a given image using a library such as OpenCV or scikit-image.
image = cv2.imread('vehicle_types\hatchback\PIC_0.jpg')
image_gray = color.rgb2gray(image) # Converting image to grayscale

features, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Input image')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG features')
plt.show()

#2. Apply your implementation to at least three different images, including both simple and complex scenes.


#3. Visualize the original image, the gradient image, and the HOG feature image.


#4. Compare the HOG features extracted from different images. Discuss the impact of varying parameters like cell size, block size, and the number of bins on the resulting HOG descriptors.