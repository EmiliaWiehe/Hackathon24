from processed_part import ProcessedPart
from processed_gripper import ProcessedGripper
from gripper_placement_optimized import GripperPlacementOptimized
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import time

# Main function
def main():
    # Load a test image
    print("Starting gripper positioning program...")
    test_image_path = r'C:\\Users\\singe\\Documents\\Desktop\\KIT\\11. Semester\\ProKI\\All Parts\\mask_20241126-142218-067.png'  # Replace with an actual path
    if os.path.exists(test_image_path):
        
        image_array = load_img(test_image_path)

        # Record the start time
        start_time = time.time()
        processed_part = ProcessedPart(image_array)
        #  Record the end time
        end_time = time.time()
        # Print the execution time
        print(f"Get part mask execution time: {end_time - start_time} seconds")

        # Predict maskh
        predicted_mask = processed_part.get_part_mask()

        com_x, com_y =   processed_part.get_part_com()

        # Open the PNG image
        gripper = Image.open(r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Johann Training Data\1\1.png').convert("RGBA")
        processed_gripper = ProcessedGripper(gripper, 2)

        collision_threshold = processed_part.get_collision_threshold()
        gripper_placement = GripperPlacementOptimized(processed_part, processed_gripper, collision_threshold)
        print(f"Collision threshold: {collision_threshold}")
        # Record the start time
        start_time = time.time()
        gripper_position = gripper_placement.determine_gripper_position_opt()
        # Record the end time
        end_time = time.time()
        # Print the execution time
        print(f"Determine position execution time: {end_time - start_time} seconds")
        
        if gripper_position is None:
            print("No valid gripper position found.")
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask, cmap='gray')  # Predicted mask
            plt.title("Predicted Mask")

            plt.subplot(1, 3, 2)
            plt.imshow(image_array)  # Original image
            plt.imshow(processed_gripper.get_resized_gripper_array(image_array.width, image_array.height,
                                                               com_x, com_y,
                                                               0), cmap='gray', alpha=0.5)  # Gripper
            plt.title("Original Image")

            plt.show()
            return
        else:
            print(f"Part center of mass: {com_x, com_y}")
            print(f"Optimal gripper position: {gripper_position}")
            g_com_x, g_com_y = gripper_position[0], gripper_position[1]
            print(f"Distance from optimal gripper position to part center of mass: {np.sqrt((com_x - g_com_x)**2 + (com_y - g_com_y)**2)}")
        
        #gripper_placement.check_gripper_position(gripper_position[0], gripper_position[1], gripper_position[2], collosion_threshold)
        
        gripper_resized_array = processed_gripper.get_resized_gripper_array(image_array.width, image_array.height,
                                                               gripper_position[0], gripper_position[1],
                                                               gripper_position[2], processed_gripper.gripper_array_unpadded)
        
        plt.subplot(1, 3, 1)
        plt.imshow( gripper_resized_array, cmap='gray')  # Combined array
        plt.title("Gripper")
        
        #print(com)

        # Visualize results
        plt.subplot(1, 3, 2)
        plt.imshow(image_array)  # Original image
        plt.imshow(gripper_resized_array, cmap='gray', alpha=0.5)  # Gripper
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        #plt.imshow(predicted_mask, cmap='gray')  # Predicted mask
        plt.imshow(processed_part.get_binary_part_mask(), cmap='gray')  # Predicted mask
        plt.title("Predicted Mask")

        plt.show()
    else:
        print(f"Error: Test image not found at {test_image_path}")


# Define functions to test bitwise XOR and AND
def test_bitwise_xor():
    np.bitwise_xor(array1, array2)

def test_bitwise_and():
    np.bitwise_and(array1, array2)

def test_addtion():
    np.left_shift(array1, array2)


def add_padding(input_array, padding_amount):
    """
    Adds padding around entries with a 1 in a 2D numpy array.

    Parameters:
    input_array (np.array): A 2D numpy array containing only 0s and 1s.
    padding_amount (int): The amount of padding to add around entries with a 1.

    Returns:
    np.array: A new 2D numpy array with the specified padding added.
    """
    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if not np.array_equal(input_array, input_array.astype(bool)):
        raise ValueError("Input array must contain only 0s and 1s.")
    if not isinstance(padding_amount, int) or padding_amount < 0:
        raise ValueError("Padding amount must be a non-negative integer.")

    # Calculate the new size of the array with padding
    rows, cols = input_array.shape
    new_rows, new_cols = rows + 2 * padding_amount, cols + 2 * padding_amount

    # Create a new array initialized with zeros
    padded_array = np.zeros((new_rows, new_cols), dtype=int)

    # Place the original array in the center of the new array
    padded_array[padding_amount:padding_amount + rows, padding_amount:padding_amount + cols] = input_array

    # Find indices where the original array has 1s
    ones_indices = np.argwhere(padded_array == 1)

    # Add padding around each 1
    for r, c in ones_indices:
        padded_array[max(0, r - padding_amount):r + padding_amount + 1,
                     max(0, c - padding_amount):c + padding_amount + 1] = 1

    return padded_array

def array_to_tf_image(array):
    """
    Converts a 2D numpy array with values between 0 and 1 to a TensorFlow RGB image.

    Parameters:
    array (np.array): A 2D numpy array with values between 0 and 1.

    Returns:
    tf.Tensor: A TensorFlow tensor representing an RGB image.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Array values must be between 0 and 1.")

    # Convert 2D array to 3D by stacking it into 3 channels (RGB)
    rgb_array = np.stack([array, array, array], axis=-1)

    # Convert to TensorFlow tensor with dtype=tf.float32
    tf_image = tf.convert_to_tensor(rgb_array, dtype=tf.float32)

    return tf_image

# Entry point
if __name__ == "__main__":
    main()