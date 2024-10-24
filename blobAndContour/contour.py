import cv2
import glob
import os
import numpy as np

# Define the path to the folder containing your images
path = "replace-with-path"
image_paths = glob.glob(os.path.join(path, "*.jpg"))

# Store contour statistics for each image
contour_statistics = []

# Iterate through each image path
for image_path in image_paths[:10]:  # Limiting to the first 10 images for demonstration
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and smooth the image (adjusted kernel size)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Reduced kernel size for less aggressive smoothing
    
    # Apply binary thresholding with a higher threshold to focus on stronger edges
    ret, binary_image = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    
    # Detect contours using RETR_TREE for more detailed hierarchy detection
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original RGB image (overlaying them)
    output_image = image.copy()
    for i, contour in enumerate(contours):
        # Generate a random color for each contour
        color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
        cv2.drawContours(output_image, [contour], -1, color, 2)  # Reduced thickness for finer contours
    
    # Display the output image with contours
    cv2.imshow("Contours", output_image)
    cv2.waitKey(0)  # Wait for a key press to move to the next image
    cv2.destroyAllWindows()
    
    # Calculate contour statistics
    num_contours = len(contours)
    contour_areas = [cv2.contourArea(c) for c in contours]
    contour_perimeters = [cv2.arcLength(c, True) for c in contours]
    
    # Store statistics in a dictionary
    contour_statistics.append({
        'image_path': image_path,
        'num_contours': num_contours,
        'contour_areas': contour_areas,
        'contour_perimeters': contour_perimeters
    })
    
    # Display the statistics for the current image
    print(f"Image: {image_path}")
    print(f"Number of Contours: {num_contours}")
    print(f"Contour Areas: {contour_areas}")
    print(f"Contour Perimeters: {contour_perimeters}\n")
