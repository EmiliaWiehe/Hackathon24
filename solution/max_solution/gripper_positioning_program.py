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


# Main function
def main():
    # Load a test image
    test_image_path = r'C:\Users\singe\Desktop\test_img_4.png'  # Replace with an actual path

    if os.path.exists(test_image_path):
        
        image_array = tf.keras.preprocessing.image.load_img(test_image_path)
        processed_part = ProcessedPart(image_array)

        # Predict mask
        predicted_mask = processed_part.get_part_mask()
        print(predicted_mask.shape)

        com_x, com_y =   processed_part.get_part_com()
        print(com_x, com_y)

        # Open the PNG image
        gripper = Image.open(r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\ProKI Hackathon 2024\Rohdaten\part_1\1.png').convert("RGBA")
        processed_gripper = ProcessedGripper(gripper)

        angle = -45
        svg_array = processed_gripper.get_resized_gripper_array(image_array.width, image_array.height, com_x, com_y, angle)
        gripper_com = processed_gripper.get_gripper_com()
        print(gripper_com)

        gripper_placement = GripperPlacement(processed_part, processed_gripper)
        collosion_threshold = 1 + processed_part.get_collision_threshold()
        print(f"Collision threshold: {collosion_threshold}")
        # Record the start time
        start_time = time.time()
        gripper_position = gripper_placement.determine_gripper_position(collosion_threshold)
        
        if gripper_position is None:
            print("No valid gripper position found.")
            return
        else:
            print(f"Optimal gripper position: {gripper_position}")
        print(gripper_placement.check_gripper_position(gripper_position[0], gripper_position[1], gripper_position[2], collosion_threshold))
        # Record the end time
        end_time = time.time()
        # Print the execution time
        print(f"Execution time: {end_time - start_time} seconds")

        plt.subplot(1, 3, 1)
        plt.imshow(gripper_placement.get_combined_array(), cmap='gray')  # Combined array
        plt.title("Gripper")
        
        #print(com)

        # Visualize results
        plt.subplot(1, 3, 2)
        plt.imshow(image_array)  # Original image
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask, cmap='gray')  # Predicted mask
        plt.title("Predicted Mask")

        plt.show()
    else:
        print(f"Error: Test image not found at {test_image_path}")



# Entry point
if __name__ == "__main__":
    main()