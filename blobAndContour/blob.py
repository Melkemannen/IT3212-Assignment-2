import cv2
import glob
import os

# Define the path to the folder containing your images
path = "replace-with-path"
image_paths = glob.glob(os.path.join(path, "*.jpg"))

# Set up the SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.maxThreshold = 100
params.filterByArea = True
params.minArea = 1
params.filterByCircularity = True
params.minCircularity = 0.3
params.filterByConvexity = True
params.minConvexity = 0.6
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Iterate through each image path
for image_path in image_paths[:10]:  # Limiting to the first 10 images for demonstration
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect blobs in the image
    keypoints = detector.detect(image)
    
    # Store statistics
    num_blobs = len(keypoints)
    blob_sizes = [kp.size for kp in keypoints]  # Sizes of each blob
    blob_positions = [kp.pt for kp in keypoints]  # (x, y) positions of each blob
    
    # Print the statistics to the terminal
    print(f"Image: {image_path}")
    print(f"Number of Blobs: {num_blobs}")
    print(f"Blob Sizes: {blob_sizes}")
    print(f"Blob Positions: {blob_positions}")
    print("\n")
    
    # Draw detected blobs as red circles on the image
    im_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Display the image with keypoints
    cv2.imshow("Blob Detection", im_with_keypoints)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
