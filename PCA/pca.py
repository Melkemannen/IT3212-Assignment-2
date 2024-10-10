import time
import cv2 
import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

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
imageMean = np.mean(images_2d, axis=0)
images_2d = images_2d - np.mean(images_2d, axis=0)
cov_matrix = np.cov(images_2d, rowvar=False)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
start_time = time.time()
eigenvalues, eigenvectors = LA.eig(cov_matrix)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

# Sort eigenvectors based on eigenvalues in descending order
vectorDict = {}
for i in range(0, len(eigenvalues)):
    vectorDict[tuple(eigenvectors[:,i])] = eigenvalues[i]

vectorDict = dict(sorted(vectorDict.items(), key=lambda item: item[1], reverse=True))

# Get the top k eigenvectors to create principal components
k = 40
PCs =  list(vectorDict.keys())[:k]
PCs = np.array(PCs)

# Project images onto lower dimensionality space (k dimensions)
reducedImages = np.dot(images_2d, PCs.T)

# Reconstruct images using principal components
reconstructedImages = np.dot(reducedImages, PCs) + imageMean

reconstructedImages = reconstructedImages.reshape(-1, 50, 50)

'''if np.iscomplexobj(reconstructedImages):
    reconstructedImages = np.real(reconstructedImages)'''

plt.figure(figsize=(12, 6))
for i in range (0,6):
    plt.subplot(2,3,i+1)
    plt.imshow(reconstructedImages[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate and plot the amount of variance contained in the principal components
variances_k = []
totalVariance = np.sum(eigenvalues)

for i in range (0, 200):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    contained_variance = np.sum(sorted_eigenvalues[:i]) / totalVariance
    variances_k.append(contained_variance)
    
plt.figure(figsize=(12, 6))
plt.plot(variances_k, label='Variance retained by k PCs')
plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Total Variance')
plt.xticks(np.arange(0, len(variances_k), 10))
plt.xlabel('k-PCs')
plt.ylabel('Variance')

plt.legend()
plt.grid()
plt.show()