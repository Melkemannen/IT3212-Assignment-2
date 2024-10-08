import time
import cv2 
import os
import numpy as np
from numpy import linalg as LA

folder_path = 'C:/Users/aless/Documents/NTNU/Datadrevet/IT3212-Assignment-2/PCA/hatchback'

# Convert images to grayscale and save

for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 300x300
    gray_image = cv2.resize(gray_image, (50, 50))
    cv2.imwrite(img_path, gray_image)


# Normalize pixel values
scaled_images = []

for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    # Read in grayscale mode for it to be a 2D array already
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Divide by 255 to get value between 0-1
    scaled_image = image / 255
    scaled_images.append(scaled_image)

# Flatten to 1D images and add to list to make 2D image list
images_2d = []

for imageArray in scaled_images:
    flattened_image  = np.array(imageArray).ravel()
    images_2d.append(flattened_image)

images_2d = np.array(images_2d)

# Calculate the covariance matrix
cov_matrix = np.cov(images_2d, rowvar=False)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
start_time = time.time()
eigenvalues, eigenvectors = LA.eig(cov_matrix)
print(eigenvalues)
print(eigenvectors)
print(time.time()-start_time)