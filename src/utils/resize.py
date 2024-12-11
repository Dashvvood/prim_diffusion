import os
from PIL import Image

def resize_images_in_folder(folder_path, size=(64, 64)):
    """
    Resize all PNG images in the specified folder and its subfolders to the given size.

    :param folder_path: Path to the folder containing images.
    :param size: Tuple specifying the new size (width, height).
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Loop through all files and subfolders in the folder
    for root, _, files in os.walk(folder_path):
        # Create a folder to save resized images, mirroring the structure
        relative_path = os.path.relpath(root, folder_path)
        output_folder = os.path.join(folder_path, "resized", relative_path)
        os.makedirs(output_folder, exist_ok=True)

        for filename in files:
            if filename.lower().endswith('.png'):
                image_path = os.path.join(root, filename)

                try:
                    with Image.open(image_path) as img:
                        # Resize the image
                        img_resized = img.resize(size)

                        # Save the resized image to the output folder
                        output_path = os.path.join(output_folder, filename)
                        img_resized.save(output_path)

                        print(f"Resized and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process '{filename}' in '{root}': {e}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing PNG images: ").strip()
    resize_images_in_folder(folder_path)
