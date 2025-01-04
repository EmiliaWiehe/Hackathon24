from processed_part import ProcessedPart
from processed_gripper import ProcessedGripper
from gripper_placement import GripperPlacement
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import tensorflow as tf
import numpy as np
import numpy as np
from PIL import Image
import cv2  # Optional for better resizing quality
import time
from collections import deque
import timeit

def get_png_file_paths(folder_path):
    """
    Get the paths of all .png files in the specified folder.

    Parameters:
        folder_path (str): The path to the folder containing .png files.

    Returns:
        list: A list of paths to .png files.
    """
    # Initialize an empty list to store file paths
    png_files = []

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the current item is a .png file
        if os.path.isfile(file_path) and file_name.lower().endswith('.png'):
            png_files.append(file_path)

    return png_files

def overlay_images(part_path, gripper_path, angle, position, output_path):
    """
    Overlays a rotated gripper image onto a part image at a specified position and saves the result.

    Parameters:
        part_path (str): Path to the part image (background).
        gripper_path (str): Path to the gripper image (to overlay).
        angle (float): Angle in degrees to rotate the gripper image.
        position (tuple): (x, y) coordinates for overlaying the gripper image on the part image.
        output_path (str): Path to save the resulting image as a .png file.

    Returns:
        None
    """
    # Open the images
    part = Image.open(part_path).convert("RGBA")
    gripper = Image.open(gripper_path).convert("RGBA")

    # Rotate the gripper image
    gripper_rotated = gripper.rotate(angle, expand=True)

    # Create a blank image for compositing
    result = Image.new("RGBA", part.size)

    # Paste the part image onto the blank result image
    result.paste(part, (0, 0))

    # Convert center of mass gripper position to top-left corner position
    position = (position[0] - gripper_rotated.width // 2, position[1] - gripper_rotated.height // 2)

    # Overlay the rotated gripper at the specified position
    result.paste(gripper_rotated, position, gripper_rotated)

    # Save the resulting image
    result.save(output_path, format="PNG")

def append_to_file(filename, content):
    """
    Appends a string to a .txt file.

    Args:
        filename (str): The name of the .txt file.
        content (str): The string to append to the file.
    """
    try:
        with open(filename, 'a') as file:
            file.write(content + '\n')  # Append the content with a newline
        print(f"Content successfully appended to {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function
def main():
    #Get the gripper
    gripper = Image.open(r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Johann Training Data\3\3.png').convert("RGBA")
    processed_gripper = ProcessedGripper(gripper, 2)

    # Path to missing_parts text file
    missing_parts_file = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Johann Training Data\3\missing_parts.txt'

    # Get all image paths in the specified folder
    folder_path = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\All Parts'
    image_paths = get_png_file_paths(folder_path)
    #image_paths = image_paths[232:]
    print(f'image_paths: {image_paths}')

    for image_path in image_paths:
        image_array = tf.keras.preprocessing.image.load_img(image_path)
        processed_part = ProcessedPart(image_array)

        collision_threshold = processed_part.get_collision_threshold()
        gripper_placement = GripperPlacement(processed_part, processed_gripper, collision_threshold)
        gripper_position = gripper_placement.determine_gripper_position()

        image_base_name = os.path.basename(image_path)
        if gripper_position is None:
            print("No valid gripper position found. For part: ", image_base_name)
            append_to_file(missing_parts_file, image_base_name)
        else:
            print(f"Optimal gripper position: {gripper_position}")
            print(f'image_base_name: {image_base_name}')
            image_output_name = f"{image_base_name[:-4]}_{gripper_position[0]}_{gripper_position[1]}_{gripper_position[2]}.png"
            output_path = os.path.join(r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Johann Training Data\3', image_output_name)
            overlay_images(image_path, r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Johann Training Data\3\3.png', gripper_position[2], (gripper_position[0], gripper_position[1]), output_path)



# Entry point
if __name__ == "__main__":
    main()