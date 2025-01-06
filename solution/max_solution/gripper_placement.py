from processed_part import ProcessedPart
from processed_gripper import ProcessedGripper
import numpy as np
from collections import deque 
import matplotlib.pyplot as plt
import time

class GripperPlacement:

    def __init__(self, processed_part, processed_gripper, collision_threshold):
        self.processed_part = processed_part
        self.processed_gripper = processed_gripper
        self.part_mask = processed_part.get_part_mask()
        self.collision_threshold = collision_threshold
        self.combined_array = None

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
        
        max_rotations = 180

        # Limit runtime to 20 seconds
        start_time = time.time()

        # Iterate through the array in radial pattern
        for index in self.radial_iterator(self.part_mask, start_index):

            # Check if the runtime exceeds 20 seconds
            if time.time() - start_time > 200:
                return None
            
            # Rotate the gripper in 5 degree steps
            for angle in range(0, max_rotations, 5):
                # Check if the gripper can be placed at the current position
                if self.check_gripper_position(index[0], index[1], angle):
                    return (index[0], index[1], angle)
            
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
    
    def radial_iterator(self, arr, start_index):
        """
        Iterate through the 2D numpy array in a radial pattern, starting from a given index.
        
        Parameters:
        arr (numpy.ndarray): The input 2D array.
        start_index (tuple): The (x, y) index to start iterating from.
        
        Yields:
        tuple: (x, y) indices of elements in the radial order.
        """
        rows, cols = arr.shape
        x0, y0 = start_index
        step_size = 4
        
        # Directions for neighbors
        d_up = (-step_size, 0)
        d_up_right = (-step_size, step_size)
        d_right = (0, step_size)
        d_down_right = (step_size, step_size)
        d_down = (step_size, 0)
        d_down_left = (step_size, -step_size)
        d_left = (0, -step_size)
        d_up_left = (-step_size, -step_size)
        directions = [d_up, d_up_right, d_right, d_down_right, d_down, d_down_left, d_left, d_up_left]

        # Queue for BFS (start with the center point)
        queue = deque([(x0, y0)])
        
        # Set to track visited indices
        visited = set()
        visited.add((x0, y0))
        
        while queue:
            x, y = queue.popleft()
            
            # Yield the current index (x, y)
            yield (x, y)
            
            # Iterate through all possible neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check if the neighbor is within bounds and not yet visited
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))