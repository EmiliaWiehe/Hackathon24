import numpy as np
from scipy import ndimage

class ProcessedGripper:
    """Processes a gripper.png to be able to retrive a np.array of the gripper as well as the center of mass.
    """

    def __init__(self, gripper):
        """Generate np.array of gripper and calculate center of mass.

        Args:
            gripper (PIL.Image): Image of the gripper.
        """
        self.gripper_array = self.gripper_conversion(gripper)
        self.gripper_com = ndimage.measurements.center_of_mass(self.gripper_array)

    def get_gripper_array(self):
        """Getter for gripper array.

        Returns:
            np.ndarray: 2D binary np.array with 1 representing the gripper geometry.
        """
        return self.gripper_array
    
    def get_gripper_com(self):
        """Getter for gripper center of mass.

        Returns:
            tupel(np.float, np.float): (x,y) - index for the gripper array representing the center of mass.
        """
        return self.gripper_com

    def gripper_conversion(self, gripper):
        """Converts a PNG file to a numpy array with binary values (0 for transparent pixels, 1 for others).

        Args:
            gripper (PIL.Image in RGBA): Image of the gripper.

        Returns:
            np.ndarray: 2D binary numpy array representing transparency in the PNG.
        """

        # Extract the alpha channel
        alpha_channel = np.array(gripper)[:, :, 3]

        # Create a binary array: 0 for transparent, 1 for others
        binary_array = (alpha_channel != 0).astype(int)

        return binary_array