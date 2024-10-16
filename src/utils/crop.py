import os
import cv2
import numpy as np
import argparse

def remove_black_borders(image):
    """
    Remove the black borders around the image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find all non-black pixels
    non_black_pixels = np.where(gray > 0)
    
    # If there's no non-black pixels, return None
    if len(non_black_pixels[0]) == 0:
        return None
    
    # Get the bounding box of the non-black pixels
    top, bottom = min(non_black_pixels[0]), max(non_black_pixels[0])
    left, right = min(non_black_pixels[1]), max(non_black_pixels[1])
    
    # Crop the image
    cropped_image = image[top:bottom+1, left:right+1]
    return cropped_image

def process_images_in_folder(folder_path, output_folder):
    """
    Traverse through all images in the given folder, remove black borders, and save the result.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image formats
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']

    # Traverse the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's an image
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_formats):
            print(f"Processing: {filename}")
            
            # Read the image
            image = cv2.imread(file_path)

            # Remove black borders
            cropped_image = remove_black_borders(image)
            
            if cropped_image is not None:
                # Save the cropped image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"Saved cropped image: {output_path}")
            else:
                print(f"No non-black content in image: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Remove black borders from images in a folder.")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help="Folder containing the images")
    parser.add_argument('-o', '--output_folder', type=str, required=True, help="Folder to save cropped images")
    args = parser.parse_args()
    
    process_images_in_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
