import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate
import math

class ProcessedGripper:
    """Processes a gripper.png to be able to retrive a np.array of the gripper as well as the center of mass.
    """

    def __init__(self, gripper, padding_amount=1):
        """Generate np.array of gripper and calculate center of mass.

        Args:
            gripper (PIL.Image): Image of the gripper.
            padding_amount (int): Amount of padding to add around the gripper.
        """
        self.gripper_array_unpadded = self.resize_for_rotation(self.gripper_conversion(gripper))
        self.gripper_array_padded = self.resize_for_rotation(
            self.add_padding(self.gripper_array_unpadded, padding_amount))

    def calc_gripper_com(self, gripper_array):
        """Calculate the center of mass for the gripper array.

        Args:
            gripper_array (np.ndarray): 2D binary numpy array representing the gripper.

        Returns:
            tuple: Tuple of floats representing the center of mass (x, y).
        """
        com_x, com_y = gripper_array.shape[1] / 2, gripper_array.shape[0] / 2
        return com_x, com_y
    
    def get_resized_gripper_array(self, image_width, image_height, index_x, index_y, angle, gripper_array=None):
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
        if gripper_array is None:
            gripper_array = self.gripper_array_unpadded
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
    
    def check_out_of_bounds(self, image_width, image_height, index_x, index_y, gripper_array):
        """Check if the gripper is out of bounds. Return all values of the
        array which are out of bounds as a set.

        Args:
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            index_x (int): x index for the gripper array representing the center of mass.
            index_y (int): y index for the gripper array representing the center of mass.
            gripper_array (np.ndarray): 2D binary numpy array representing the gripper.

        Returns:
            set: Set of all values of the array which are out of bounds.
        """
        start_x = index_x - int(self.gripper_com[0])
        start_y = index_y - int(self.gripper_com[1])
        end_x = start_x + gripper_array.shape[1]
        end_y = start_y + gripper_array.shape[0]

        out_of_bounds = set()
        if start_x < 0:
            out_of_bounds.update(gripper_array[:, :abs(start_x)].flatten())
        if start_y < 0:
            out_of_bounds.update(gripper_array[:abs(start_y), :].flatten())
        if end_x > image_width:
            out_of_bounds.update(gripper_array[:, -(end_x - image_width):].flatten())
        if end_y > image_height:
            out_of_bounds.update(gripper_array[-(end_y - image_height):, :].flatten())

        return out_of_bounds

    def place_gripper(self, gripper_array, image_width, image_height, index_x, index_y):
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
        com_x, com_y = gripper_array.shape[1] / 2, gripper_array.shape[0] / 2
        start_x = index_x - int(com_x)
        start_y = index_y - int(com_y)
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
    
    def get_binary_encoded_rotation_array(self, gripper_array, initial_power=1, rotation_step=8, total_rotation=360):
        """
        Rotates the gripper array by increments. Each increment is encoded as a power of two.

        Args:
            gripper_array (np.ndarray): Square array, resized for rotation with only zeros and ones.
            initial_power (int): The starting power for scaling ones (default is 1).
            rotation_step (int): The step size for rotation in degrees (default is 8).
            total_rotation (int): Total rotation in degrees (default is 360).

        Returns:
            np.ndarray: The resulting array after all rotations and bitwise OR operations.
        """

        # Ensure the array is of the correct size and binary
        assert np.array_equal(gripper_array, gripper_array.astype(bool)), "Array must contain only 0s and 1s."

        result_array = np.zeros_like(gripper_array, dtype=np.int64)

        num_steps = total_rotation // rotation_step
        for step in range(num_steps):
            # Rotate the array
            rotated_array = rotate(gripper_array, angle=rotation_step * (step + 1), reshape=False, order=1)

            # Threshold rotated array to make it binary
            rotated_array = (rotated_array >= 0.5).astype(np.int64)

            # Scale the ones by the current power of two
            scaled_array = rotated_array * (2 ** (initial_power + step))

            # Accumulate the result
            result_array = np.bitwise_or(result_array, scaled_array)

        # Add the original array to the result
        result_array = np.bitwise_or(result_array, gripper_array * (2 ** initial_power))

        return result_array
    
    def add_padding(self, input_array, padding_amount):
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