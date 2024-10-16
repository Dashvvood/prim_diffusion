import os
import cv2
import numpy as np
import argparse

def calculate_mean_and_variance(image):
    """
    Calculate the mean and variance of a grayscale image.
    """
    # Convert image to float to avoid overflow
    image = image.astype(np.float32)

    # Calculate mean and variance for the grayscale image
    mean = np.mean(image)
    variance = np.var(image)
    return mean, variance

def process_images_in_folder(folder_path):
    """
    Traverse all grayscale images in the folder, calculate the mean and variance of each image,
    and compute the overall mean and variance.
    """
    total_mean = 0
    total_variance = 0
    image_count = 0

    # Supported image formats
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']

    # Traverse the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's an image
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_formats):
            print(f"Processing: {filename}")

            # Read the image in grayscale mode
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Calculate the mean and variance for the grayscale image
                mean, variance = calculate_mean_and_variance(image)

                # Accumulate the total mean and variance
                total_mean += mean
                total_variance += variance
                image_count += 1

    if image_count > 0:
        # Calculate the average mean and variance for all images in the folder
        avg_mean = total_mean / image_count
        avg_variance = total_variance / image_count

        print(f"Number of images in the folder: {image_count}")
        print(f"Overall pixel mean: {avg_mean}")
        print(f"Overall pixel variance: {avg_variance}")
    else:
        print("No images found in the folder.")

# Usage
def main():
    parser = argparse.ArgumentParser(description='Process images in a folder to calculate mean and variance.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    process_images_in_folder(args.input)

if __name__ == "__main__":
    main()
