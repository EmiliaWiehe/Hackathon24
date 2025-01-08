from tensorflow.keras import models, preprocessing
import tensorflow as tf
from scipy import ndimage
import numpy as np

class ProcessedPart:

    def __init__(self, part_path):
        """Starts the calculation of the part mask and center of mass.

        Args:
            part (tf.keras.image): Part to process.
        """
        part = tf.keras.preprocessing.image.load_img(part_path)
        self.part_mask = self.ml_hole_localization(part)
        self.part_com = self.calc_part_com()
        self.collision_threshold = self.calc_collision_threshold(self.part_mask)

    def get_part_mask(self):
        """Getter for the part mask.

        Returns:
            np.array: The part mask as an array with the size of the original part. Closer to 1 = likely a hole.
        """
        return self.part_mask
    
    def get_part_com(self):
        """Getter for the part center of mass.

        Returns:
            tupel(np.float, np.float): (x,y) - index for the part mask representing the center of mass.
        """
        return self.part_com
    
    def get_collision_threshold(self):
        """Getter for the collision threshold.

        Returns:
            float: The collision threshold for the gripper.
        """
        return self.collision_threshold
    
    def calc_part_com(self):
        """Calculate the center of mass of the part as a tuple of integers.

        Returns:
            tupel(int, int): Index for the part array representing the center of mass.
            First index is the x index, second index is the y index.
        """
        part = self.invert_image(self.part_mask) #Holes should be 0, non-holes should be 1 for com calc.
        part_com_float = ndimage.measurements.center_of_mass(part)
        part_com = tuple(int(round(x)) for x in part_com_float)

        # Reorder the tuple so that first index ist x index and second index is y index.
        part_com_reordered = (part_com[1], part_com[0])
        return part_com_reordered
    
    def invert_image(self, array):
        """Inverts a 2D NumPy array with values between 0 and 1.

        Args:
            array (np.ndarray): Input 2D array with values between 0 and 1.

        Returns:
            np.ndarray: Inverted 2D array where each value `x` is replaced by `1 - x`.
        """
        if not (np.all(array >= 0) and np.all(array <= 1)):
            raise ValueError("All values in the array must be between 0 and 1.")

        return 1 - array

    def ml_hole_localization(self, part):
        """Uses binary semantic segmentation to classify each pixel of "part" as a hole (1) or non-hole (0).

        Args:
            part (tf.keras.image): Colored Image of a part with holes (256x256 pixels).

        Returns:
            np.array [256, 256]: 256x256 2D np.array with values between 0 and 1. Values closer to 1 are more likely to be holes.
        """

        model_dir = "./solution/model/mask_model_small.keras"
        model = models.load_model(model_dir)

        normalized_part = preprocessing.image.img_to_array(part) / 255.0 #Normalize array values between 0 and 1
        part_outline = self.missing_pixels_to_array_tf(part) #Get a mask of missing pixels (NaN values)

        # Save Original img dimensions
        original_height, original_width = normalized_part.shape[:2]

        normalized_part = tf.image.resize(normalized_part, [256, 256]) #model only accepts 256x256 images
        normalized_part = tf.expand_dims(normalized_part, axis=0)  # Add batch dimension

        predicted_mask = model.predict(normalized_part)
        predicted_mask_original_size = tf.image.resize(predicted_mask, [original_height, original_width])
        predicted_mask_original_size = predicted_mask_original_size[0,:,:,0] #Trim additional dimensions
        predicted_mask_original_size = predicted_mask_original_size.numpy() #Convert tf.image to np.array
        self.max_mask_value = np.max(predicted_mask_original_size) #Get the maximum value in the mask
        predicted_mask_original_size = predicted_mask_original_size + part_outline #Add missing pixels to the mask
        predicted_mask_original_size = np.minimum(predicted_mask_original_size, self.max_mask_value) #Clip values to max_mask_value

        return predicted_mask_original_size
    
    def missing_pixels_to_array_tf(self, image):
        """Converts a tf.image (tf.Tensor) to a 2D NumPy array where missing pixels
        (pixels with all channel values equal to 0) are represented by 1,
        and all other pixels by 0.

        Args:
            image (tf.Tensor): Input tf.image (a tensor representing an image).

        Returns:
            numpy.ndarray: A 2D NumPy array with 1 for missing pixels and 0 otherwise.
        """
        # Ensure the input is a TensorFlow tensor
        image = tf.convert_to_tensor(image)
        
        # Identify missing pixels (all channels are zero for a pixel)
        missing_mask = tf.reduce_all(tf.equal(image, 0), axis=-1)
        
        # Convert the missing pixel mask to integer values (1 for missing, 0 for non-missing)
        missing_mask = tf.cast(missing_mask, tf.int32)
        
        # Convert the result to a NumPy array
        result_array = missing_mask.numpy()
        
        return result_array
    
    def calc_collision_threshold(self, part_mask):
        """Calculates the collision threshold for the gripper to avoid collisions with the part.
        Values of exactly 1 will not count towards the threshold to compensate for missing pixels.

        Args:
            part_mask (np.array): 2D np.array with values between 0 and 1. Closer to 1 = likely a hole.

        Returns:
            float: The collision threshold for the gripper. A value between 0 and 1. If there is
            a higher probability for holes across the entire part, the threashold will also be higher.
        """
        # Calculate the number of pixels in the part mask
        num_pixels = part_mask.size

        # Replace all values of exactly 1 with 0 to compensate for missing pixels
        part_mask = np.where(part_mask == self.max_mask_value, 0, part_mask)
        
        # Calculate the number of hole pixels
        num_holes = np.sum(part_mask)

        # Subtract the number of values with exactly 0 from num_pixels
        num_zeros = np.sum(part_mask == 0)
        adjusted_num_pixels = num_pixels - num_zeros
    
        
        # Calculate the collision threshold (1.2 is a experimentally determined factor)
        collision_threshold = 1.2 * num_holes / adjusted_num_pixels
        
        return collision_threshold