from processed_part import ProcessedPart
from processed_gripper import ProcessedGripper
import numpy as np
from collections import deque 
import matplotlib.pyplot as plt
import time
from scipy.ndimage import rotate

class GripperPlacementOptimized:

    def __init__(self, processed_part, processed_gripper, collision_threshold):
        self.processed_part = processed_part
        self.processed_gripper = processed_gripper

        # Get half the width of the padded gripper array
        half_gripper_width = processed_gripper.gripper_array_padded.shape[1] // 2
        
        # Add this half width as padding to the binary part mask
        self.resized_binary_part_mask = np.pad(processed_part.get_binary_part_mask(), 
            pad_width=half_gripper_width, mode='constant', constant_values=1)
        # Note the offset for the indexes incured by resizing the part mask
        self.offset = half_gripper_width
    

        
    def print_binary_int64(self, num):
        """
        Prints a NumPy int64 number as a binary number.

        Parameters:
        num (np.int64): The 64-bit integer to be printed in binary.
        """
        if not isinstance(num, np.int64):
            raise TypeError("Input must be a numpy int64 type.")

        # Use NumPy's functions to handle 64-bit numbers correctly
        # Convert the number to its unsigned 64-bit equivalent
        as_unsigned = np.uint64(num)
        
        # Format as a 64-bit binary string
        binary_representation = f"{as_unsigned:064b}"

        print(binary_representation)
        
    def check_all_gripper_rotations(self, x, y, rotation_steps=8, total_rotation=360):
        binary_part_mask = self.resized_binary_part_mask.astype(np.int64)
        # Multiply each element of binary_part_mask by 2^62
        binary_part_mask = binary_part_mask * 2**62

        binary_part_mask_height, binary_part_mask_width = binary_part_mask.shape

        overlayed_rotations = self.processed_gripper.get_binary_encoded_rotation_array(
            self.processed_gripper.gripper_array_padded, initial_power=1, 
            rotation_step=rotation_steps, total_rotation=total_rotation)
        
        
        overlayed_rotations_placed = self.processed_gripper.place_gripper(
            overlayed_rotations, binary_part_mask_width, binary_part_mask_height, x, y)
        overlayed_rotations_placed = overlayed_rotations_placed.astype(np.int64)

        part_and_rotations = np.bitwise_or(overlayed_rotations_placed, binary_part_mask)
        part_and_rotations = part_and_rotations.astype(np.int64)


        # Get all unique elements in the array which are greater than 2^62
        # This is equivalent to getting all rotations which have collided with a hole.
        unique_elements = self.get_element_list(part_and_rotations)

        # Subtract 2^62 from all elemets of unique_elemets
        unique_elements = {element - 2**62 for element in unique_elements} 

        binary_number = self.bitwise_or_on_set(unique_elements).astype(np.int64)

        encoded_rotation = self.get_powers_not_in_number(binary_number, total_rotation // rotation_steps)

        # if the encoded rotation is not empty, return the first element
        if encoded_rotation:
            angle = encoded_rotation.pop() * rotation_steps
            return angle
        
        return None

    def bitwise_or_on_set(self, np_int64_set):
        """
        Perform a bitwise OR operation on all elements of a set of NumPy 64-bit integers.

        Args:
            np_int64_set (set): A set of np.int64 integers.

        Returns:
            np.int64: The result of the bitwise OR operation on all elements in the set.

        Raises:
            ValueError: If the set is empty.
        """
        if not np_int64_set:
            raise ValueError("The input set is empty. Please provide a set with at least one element.")

        # Initialize result with 0 (neutral element for bitwise OR)
        result = np.int64(0)

        # Perform bitwise OR for all elements
        for num in np_int64_set:
            result |= num

        return result

    def get_powers_not_in_number(self, number, max_power=63):
        """Returns a set of all powers of two not present in the input number."""
        return {i for i in range(1, max_power) if not number & 2**i}

    def get_element_list(self, array):
        """Returns a set of all unique elements in the input array which
        are greater than 2^62."""
        return set(array[array > 2**62])
    
    def get_resized_part_com(self):
        """Returns the center of mass in the coordinates of the resized part array."""
        return (self.processed_part.get_part_com()[0] + self.offset, 
            self.processed_part.get_part_com()[1] + self.offset)
    
    def determine_gripper_position_opt(self):
        """Determines the optimal gripper position to avoid collisions with the part.
        
        The function iterates over all possible gripper positions and angles to find
        the optimal placement that avoids collisions with the part. The gripper is placed
        at the position with the highest possible threshold value.
        
        Returns:
        tuple: A tuple containing the best x, y, and angle values for the gripper.
        """
        # Get all valid gripper positions
        #placable_gripper_positions = self.get_placeable_gripper_positions()

        # Set the part mask com as the starting position
        start_index = self.get_resized_part_com()

        # Limit runtime to 20 seconds
        start_time = time.time()

        counter = 0

        # Iterate through the array in radial pattern
        for index in self.radial_iterator(self.resized_binary_part_mask, start_index, 4):
            counter += 1

            # Check if the runtime exceeds 20 seconds
            if time.time() - start_time > 200:
                return None
            
            try:
                valid_angle = self.check_all_gripper_rotations(index[0], index[1])
            except ValueError:
                return None
            # If valid angle is not None, return the position and angle
            if valid_angle is not None:
                print(f"Counter: {counter}")
                return (index[0] - self.offset, index[1] - self.offset, valid_angle)
            
        return None
    

    
    def radial_iterator(self, arr, start_index, step_size):
        """
        Radial iterator that yields elements in a 2D array in a radial order starting from a given index.
        The step size doubles every second iteration.

        Args:
            arr (np.ndarray): 2D array to iterate over.
            start_index (tuple): (x, y) starting index for the iteration.
            step_size (int): The initial step size for the radial iterator.

        Yields:
            tuple: (x, y) indices of elements in the radial order.
        """
        rows, cols = arr.shape
        x0, y0 = start_index

        print(f"Start index: {start_index}")

        # Directions for neighbors
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        current_step_size = step_size
        iteration = 0
        current_x, current_y = x0, y0

        while True:
            for direction in directions:
                for step in range(1, current_step_size + 1):
                    x = current_x + direction[0] * step
                    y = current_y + direction[1] * step
                    if 0 <= x < rows and 0 <= y < cols:
                        yield (x, y)

            iteration += 1
            if iteration % 5 == 0:
                current_step_size += 2
                print(f"Current step size: {current_step_size}")
            current_x, current_y = x, y