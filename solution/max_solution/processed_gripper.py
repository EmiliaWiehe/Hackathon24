import numpy as np
from scipy import ndimage
import math

class ProcessedGripper:
    """Processes a gripper.png to be able to retrive a np.array of the gripper as well as the center of mass.
    """

    def __init__(self, gripper):
        """Generate np.array of gripper and calculate center of mass.

        Args:
            gripper (PIL.Image): Image of the gripper.
        """
        self.gripper_array = self.gripper_conversion(gripper)
        self.gripper_com = self.calc_gripper_com(self.gripper_array)

    def calc_gripper_com(self, gripper_array):
        """Calculate the center of mass of the gripper as a tuple of integers.

        Args:
            gripper_array (np.ndarray): 2D binary numpy array representing the gripper.

        Returns:
            tupel(int, int): Index for the gripper array representing the center of mass.
            First index is the x index, second index is the y index.
        """
        gripper = self.invert_image(gripper_array)  # Gripper should be 0, non-gripper should be 1 for com calc.
        gripper_com_float = ndimage.measurements.center_of_mass(gripper)
        print(gripper_com_float)
        gripper_com = tuple(int(round(x)) for x in gripper_com_float)

        # Reorder the tuple so that first index ist x index and second index is y index.
        gripper_com_reordered = (gripper_com[1], gripper_com[0])
        return gripper_com_reordered
    
    def invert_image(self, array):
        """
        Inverts a 2D NumPy array with values between 0 and 1.

        Args:
            array (np.ndarray): Input 2D array with values between 0 and 1.

        Returns:
            np.ndarray: Inverted 2D array where each value `x` is replaced by `1 - x`.
        """
        if not (np.all(array >= 0) and np.all(array <= 1)):
            raise ValueError("All values in the array must be between 0 and 1.")

        return 1 - array

    def get_gripper_array(self):
        """Getter for gripper array.

        Returns:
            np.ndarray: 2D binary np.array with 1 representing the gripper geometry.
        """
        return self.gripper_array
    
    def get_resized_gripper_array(self, image_width, image_height, index_x, index_y, angle):
        """Getter for resized gripper array. Resizes the gripper array to the given image dimensions,
        places the center of mass at the given index and rotates the gripper by the given angle.

        Args:
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            index_x (int): x index for the gripper array representing the center of mass.
            index_y (int): y index for the gripper array representing the center of mass.
            angle (int): Angle of the gripper.

        Returns:
            np.ndarray: 2D binary np.array with 1 representing the gripper geometry.
        """
        gripper_array = self.gripper_array
        gripper_array = self.resize_for_rotation(gripper_array)
        gripper_array = self.rotate_image(gripper_array, angle)
        gripper_array = self.resize_to_contain_ones(gripper_array)
        gripper_array = self.resize_gripper(gripper_array, image_width, image_height, index_x, index_y)
        return gripper_array
    
    def resize_for_rotation(self, array):
        """
        Adjusts the size of a 2D array to ensure it can be rotated 
        without cutting off any part of the original array.

        Args:
            array (numpy.ndarray): Input 2D array.

        Returns:
            numpy.ndarray: Resized 2D array with padding.
        """
        # Get the original dimensions
        rows, cols = array.shape
        
        # Calculate the diagonal length (hypotenuse)
        diagonal = math.sqrt(rows**2 + cols**2)
        new_size = math.ceil(diagonal)
        
        # Ensure the new dimensions are odd to center the array easily
        new_size = new_size + (new_size % 2)
        
        # Calculate padding for each side
        pad_top = (new_size - rows) // 2
        pad_bottom = new_size - rows - pad_top
        pad_left = (new_size - cols) // 2
        pad_right = new_size - cols - pad_left
        
        # Pad the array with zeros
        resized_array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return resized_array
    
    def resize_to_contain_ones(self, array):
        """Resizes a 2D NumPy array to the smallest possible array that contains 
        all indexes with value 1.
        
        Args:
            array (numpy.ndarray): Input 2D array with 0s and 1s.

        Returns:
            numpy.ndarray: Resized 2D array.
        """
        # Find the indexes of all 1s
        ones_indexes = np.argwhere(array == 1)
        
        if ones_indexes.size == 0:  # If no 1s, return an empty array
            return np.array([[]])
        
        # Determine the bounds for slicing
        min_row, max_row = ones_indexes[:, 0].min(), ones_indexes[:, 0].max()
        min_col, max_col = ones_indexes[:, 1].min(), ones_indexes[:, 1].max()
        
        # Slice the array to include only the relevant portion
        resized_array = array[min_row:max_row + 1, min_col:max_col + 1]
        
        return resized_array
    
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
    
    def rotate_image(self, image, angle):
        """Rotates the image by the given angle.

        Args:
            image (np.ndarray): 2D binary numpy array representing the image.
            angle (int): Angle to rotate the image.

        Returns:
            np.ndarray: Rotated 2D binary numpy array.
        """
        return ndimage.rotate(image, angle, reshape=False, order=0)

    def resize_gripper(self, gripper_array, image_width, image_height, index_x, index_y):
        """Resizes the gripper array to the given image dimensions and places the center of mass at the given index.

        Args:
            gripper_array (np.ndarray): 2D binary numpy array representing the gripper.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            index_x (int): x index for the gripper array representing the center of mass.
            index_y (int): y index for the gripper array representing the center of mass.

        Returns:
            np.ndarray: Resized 2D binary numpy array.
        """
        resized_gripper = np.zeros((image_height, image_width))
        com_x, com_y = self.calc_gripper_com(gripper_array)
        start_x = index_x - int(com_x)
        start_y = index_y - int(com_y)
        print(gripper_array.shape)
        end_x = start_x + gripper_array.shape[1]
        end_y = start_y + gripper_array.shape[0]

        if start_x < 0:
            raise ValueError("Gripper array cannot be placed at the given index: start_x is out of bounds.")
        if start_y < 0:
            raise ValueError("Gripper array cannot be placed at the given index: start_y is out of bounds.")
        if end_x > image_width:
            raise ValueError("Gripper array cannot be placed at the given index: end_x exceeds image width.")
        if end_y > image_height:
            raise ValueError("Gripper array cannot be placed at the given index: end_y exceeds image height.")

        resized_gripper[start_y:end_y, start_x:end_x] = gripper_array
        return resized_gripper