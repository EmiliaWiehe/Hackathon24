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
        self.part_mask = processed_part.get_part_mask()
        self.collision_threshold = collision_threshold
        self.combined_array = None
        self.rotation_step = 5
        self.radial_iterator_step = 4

    def check_gripper_position(self, x, y, angle):
        """Checks if the gripper can be placed at the given position without colliding with the part.
        To do so, the gripper_array and part_mask are added together. If the sum is greater than the 
        collision_threshold, the gripper is colliding with the part.

        Args:
            x (int): x index for the gripper array representing the center of mass.
            y (int): y index for the gripper array representing the center of mass.
            angle (int): Angle of the gripper.
            collision_threshold (float): Threshold for collision detection. 1 <= collision_threshold <= 2.
            A threashold of 1 means there will always be a collision detected, a threshold of 2 means there will 
            never be a collision detected.

        Returns:
            bool: True if the gripper can be placed at the given position without colliding with the part.
        """
        # Get the resized gripper array
        try:
            gripper_array = self.processed_gripper.get_resized_gripper_array(
                self.part_mask.shape[1], self.part_mask.shape[0], x, y, angle
            )
        except ValueError:
            return False

        # Add the gripper array and the part mask
        self.combined_array = gripper_array + self.part_mask

        # Check if the sum exceeds the collision threshold
        if np.any(self.combined_array > 1 + self.collision_threshold):
            return False

        return True
        
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
        binary_part_mask = self.processed_part.get_binary_part_mask()
        binary_part_mask = binary_part_mask.astype(np.int64)
        # Multiply each element of binary_part_mask by 2^62
        binary_part_mask = binary_part_mask * 2**62

        binary_part_mask_width, binary_part_mask_height = binary_part_mask.shape


        overlayed_rotations = self.get_overlayed_rotations_gripper_array(
            initial_power=1, rotation_step=rotation_steps, total_rotation=total_rotation)
        overlayed_rotations_placed = self.processed_gripper.place_gripper(
            overlayed_rotations, binary_part_mask_height, binary_part_mask_width, x, y)
        overlayed_rotations_placed = overlayed_rotations_placed.astype(np.int64)

        part_and_rotations = overlayed_rotations_placed + binary_part_mask
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
            return encoded_rotation.pop() * rotation_steps
        
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

    def get_overlayed_rotations_gripper_array(self, initial_power=1, rotation_step=8, total_rotation=360):
        # Get the gripper array
        gripper_array = self.processed_gripper.get_gripper_array()
        rotateable_gripper_array = self.processed_gripper.resize_for_rotation(gripper_array)

        # Ensure the array is of the correct size and binary
        assert np.array_equal(rotateable_gripper_array, rotateable_gripper_array.astype(bool)), "Array must contain only 0s and 1s."

        result_array = np.zeros_like(rotateable_gripper_array, dtype=np.int64)

        num_steps = total_rotation // rotation_step
        for step in range(num_steps):
            # Rotate the array
            rotated_array = rotate(rotateable_gripper_array, angle=rotation_step * (step + 1), reshape=False, order=1)

            # Threshold rotated array to make it binary
            rotated_array = (rotated_array >= 0.5).astype(np.int64)

            # Scale the ones by the current power of two
            scaled_array = rotated_array * (2 ** (initial_power + step))

            # Accumulate the result
            result_array = np.bitwise_or(result_array, scaled_array)

        # Add the original array to the result
        result_array = np.bitwise_or(result_array, rotateable_gripper_array * (2 ** initial_power))

        return result_array

    def determine_gripper_position(self):
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
        start_index = self.processed_part.get_part_com()

        # Set the max rotations based on the gripper symmetry
        match self.processed_gripper.get_symetric_axes():
            case 0:
                max_rotations = 360
            case 1:
                max_rotations = 180
            case 2:
                max_rotations = 90
            case _:
                max_rotations = 360
        
        max_rotations = self.rotation_step

        # Limit runtime to 20 seconds
        start_time = time.time()

        # Iterate through the array in radial pattern
        for index in self.radial_iterator(self.part_mask, start_index, self.radial_iterator_step):

            # Check if the runtime exceeds 20 seconds
            if time.time() - start_time > 200:
                return None
            
            # Rotate the gripper in 5 degree steps
            for angle in range(0, max_rotations, self.rotation_step):
                print(f"Checking position {index} with angle {angle}")
                # Check if the gripper can be placed at the current position
                if self.check_gripper_position(index[0], index[1], angle):
                    return (index[0], index[1], angle)
            
        return None
    
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
        start_index = self.processed_part.get_part_com()

        # Limit runtime to 20 seconds
        start_time = time.time()

        # Iterate through the array in radial pattern
        for index in self.radial_iterator(self.part_mask, start_index, self.radial_iterator_step):

            # Check if the runtime exceeds 20 seconds
            if time.time() - start_time > 200:
                return None
            
            try:
                valid_angle = self.check_all_gripper_rotations(index[0], index[1])
            except ValueError:
                return None
            # If valid angle is not None, return the position and angle
            if valid_angle is not None:
                return (index[0], index[1], valid_angle)
            
        return None
    
    def get_placeable_gripper_positions(self):
        """Uses the smallest_square_array method on the gripper to get all x and y positions for the gripper 
        which are within the bounds of the part mask. The x and y positions represent the center of mass 
        of the gripper.
        
        Returns: 
            np.array: 2D numpy array with all possible x and y positions for the gripper.
        """
        
        # Get the gripper array
        gripper_array = self.processed_gripper.get_gripper_array()
        
        # Get the smallest square array of the gripper
        square_gripper = self.smallest_square_array(gripper_array)
        
        # Get the dimensions of the gripper
        rows, cols = square_gripper.shape
        
        # Get the dimensions of the part mask
        part_rows, part_cols = self.part_mask.shape

        # Get gripper center of mass
        g_com_x, g_com_y = self.processed_gripper.get_gripper_com()
        
        # Initialize the list of placeable gripper positions
        placeable_positions = []
        
        # Iterate over all possible positions
        for x in range(part_cols - cols + 1):
            for y in range(part_rows - rows + 1):
                placeable_positions.append((x + g_com_x, y + g_com_y))
        
        return np.array(placeable_positions)
    
    def smallest_square_array(cls, arr):
        """
        Squares a 2D NumPy array by cropping it to a square shape.
        
        The resulting array will have dimensions equal to the smaller of the
        two dimensions of the input array (rows or columns).
        
        Args:
        arr (numpy.ndarray): The input 2D array which may not be square.
        
        Returns:
        numpy.ndarray: A square sub-array with dimensions equal to the smaller
                       of the input array's dimensions.
        """
        # Get the dimensions of the input array
        rows, cols = arr.shape
        
        # Find the size of the smaller dimension
        size = min(rows, cols)
        
        # Crop the array to the square size
        square_arr = arr[:size, :size]
        
        return square_arr
    
    def get_combined_array(self):
        """Getter for the combined array.

        Returns:
            np.ndarray: 2D np.array with the gripper and part mask added together.
        """
        return self.combined_array
    
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