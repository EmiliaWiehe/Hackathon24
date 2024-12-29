from processed_part import ProcessedPart
from processed_gripper import ProcessedGripper
import numpy as np

class GripperPlacement:

    def __init__(self, processed_part, processed_gripper):
        self.processed_part = processed_part
        self.processed_gripper = processed_gripper
        self.combined_array = None

    def check_gripper_position(self, x, y, angle, collision_threshold):
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

        # Get the part mask
        part_mask = self.processed_part.get_part_mask()

        # Get the resized gripper array
        gripper_array = self.processed_gripper.get_resized_gripper_array(
            part_mask.shape[1], part_mask.shape[0], x, y, angle
        )

        # Add the gripper array and the part mask
        self.combined_array = gripper_array + part_mask

        # Check if the sum exceeds the collision threshold
        if np.any(self.combined_array > collision_threshold):
            return False

        return True
    
    def get_combined_array(self):
        """Getter for the combined array.

        Returns:
            np.ndarray: 2D np.array with the gripper and part mask added together.
        """
        return self.combined_array