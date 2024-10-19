import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#Dataset used: Vehicle type recognition dataset
folder_path = "C:/Users/morom/Documents/git repos/datadrevet/assignment 2/IT3212-Assignment-2/Fourier/r7bthvstxw-1/hatchback"
    
#load images to grayscale, save
#imageList = []
#for filename in os.listdir(folder_path):
#    img_path = os.path.join(folder_path, filename)
#    image = cv2.imread(img_path)
#    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Don't resize for now
    

#1. 
#Load a grayscale image and apply the 2D Discrete Fourier Transform (DFT) to it. Visualize
#the original image and its frequency spectrum (magnitude). Submit the images, and
#explanation.
single_image = cv2.imread("Fourier/r7bthvstxw-1/hatchback/PIC_0.jpg", cv2.IMREAD_GRAYSCALE)
#Using https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
# and https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
#As a source here, making some changes where needed
#DFT, assuming we are allowed to use cv2.dft :))) 
dft = cv2.dft(np.float32(single_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
#Compute the frequency spectrum (magnitude)
mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(single_image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(mag_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()
#We get a pretty bad resolution, but the point comes across, rescale as needed

#2. 
# Implement a low-pass filter in the frequency domain to remove high-frequency noise from
# an image. Compare the filtered image with the original image. Submit images, and analysis
# of the results.


#3. 
# Implement a high-pass filter to enhance the edges in an image. Visualize the filtered image
# and discuss the effects observed. Submit images, and explanation.


#4. 
# Implement an image compression technique using Fourier Transform by selectively keeping
# only a certain percentage of the Fourier coefficients. Evaluate the quality of the
# reconstructed image as you vary the percentage of coefficients used. Submit the images,
# and your observations on image quality and compression ratio.

#Here i will submit the entire hatchback folder as an example

