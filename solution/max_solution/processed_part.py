from tensorflow.keras import models, preprocessing
import tensorflow as tf
from scipy import ndimage

class ProcessedPart:
    """Processes a part.png to be able to retrive a mask of the hole positions as well as the center of mass.
    """
    def __init__(self, part):
        """Starts the calculation of the part mask and center of mass.

        Args:
            part (tf.keras.image): Part to process.
        """
        self.part_mask = self.ml_hole_localization(part)
        self.part_com = self.calc_part_com()

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
    
    def calc_part_com(self):
        """Calculate the center of mass of the part as a tuple of integers.

        Returns:
            tupel(int, int): Index for the part array representing the center of mass.
            First index is the x index, second index is the y index.
        """
        part_com_float = ndimage.measurements.center_of_mass(self.part_mask)
        part_com = tuple(int(round(x)) for x in part_com_float)

        # Reorder the tuple so that first index ist x index and second index is y index.
        part_com_reordered = (part_com[1], part_com[0])
        return part_com_reordered

    def ml_hole_localization(self, part):
        """Uses binary semantic segmentation to classify each pixel of "part" as a hole (1) or non-hole (0).

        Args:
            part (tf.keras.image): Colored Image of a part with holes (256x256 pixels).

        Returns:
            np.array [256, 256]: 256x256 2D np.array with values between 0 and 1. Values closer to 1 are more likely to be holes.
        """

        model_dir = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Program Files\model\model_3.keras'
        model = models.load_model(model_dir)

        normalized_part = preprocessing.image.img_to_array(part) / 255.0 #Normalize array values between 0 and 1
            
        # Save Original img dimensions
        original_height, original_width = normalized_part.shape[:2]

        normalized_part = tf.image.resize(normalized_part, [256, 256]) #model only accepts 256x256 images
        normalized_part = tf.expand_dims(normalized_part, axis=0)  # Add batch dimension

        predicted_mask = model.predict(normalized_part)
        predicted_mask_original_size = tf.image.resize(predicted_mask, [original_height, original_width])
        predicted_mask_original_size = predicted_mask_original_size[0,:,:,0] #Trim additional dimensions
        predicted_mask_original_size = predicted_mask_original_size.numpy() #Convert tf.image to np.array

        return predicted_mask_original_size