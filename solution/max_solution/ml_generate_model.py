import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os

# Define U-Net model
def unet_model(input_shape=(128, 128, 3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)

    # Bottleneck
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(bottleneck)

    # Decoder
    up3 = layers.Conv2DTranspose(256, 2, strides=2, activation='relu', padding='same')(bottleneck)
    up3 = layers.concatenate([up3, conv3])
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(up3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)

    up2 = layers.Conv2DTranspose(128, 2, strides=2, activation='relu', padding='same')(conv4)
    up2 = layers.concatenate([up2, conv2])
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(up2)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)

    up1 = layers.Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')(conv5)
    up1 = layers.concatenate([up1, conv1])
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv6)

    return models.Model(inputs, outputs)

class ImageMaskGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, target_size=(256, 256), shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        # Get list of image and mask filenames
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])


        # Make sure the number of images and masks match
        assert len(self.image_filenames) == len(self.mask_filenames), "Number of images and masks must be the same."

        self.indexes = np.arange(len(self.image_filenames))
        self.on_epoch_end()

    def __len__(self):
        # The number of batches per epoch
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        # Get indices of the batch
        batch_indices = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Get the corresponding image and mask filenames
        batch_image_filenames = [self.image_filenames[i] for i in batch_indices]
        batch_mask_filenames = [self.mask_filenames[i] for i in batch_indices]

        # Load and process the images and masks
        images = np.array([self.load_image(img) for img in batch_image_filenames])
        masks = np.array([self.load_mask(mask) for mask in batch_mask_filenames])

        return images, masks

    def load_image(self, filename):
        # Load and preprocess the image
        img_path = os.path.join(self.image_dir, filename)
        img = load_img(img_path, target_size=self.target_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        return img

    def load_mask(self, filename):
        # Load and preprocess the mask
        mask_path = os.path.join(self.mask_dir, filename)
        mask = load_img(mask_path, target_size=self.target_size, color_mode='grayscale')
        mask = img_to_array(mask) / 255.0  # Normalize to [0, 1] or [0, 255] if binary mask
        return mask

    def on_epoch_end(self):
        # Shuffle the indices after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Main function
def main():
    # Path to save the model to
    model_dir = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Program Files\model\model_4.keras'
    # Define paths to your data
    image_dir = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Program Files\part_2_img\class1'
    mask_dir = r'C:\Users\singe\Documents\Desktop\KIT\11. Semester\ProKI\Program Files\part_2_mask\class1'

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: Directories {image_dir} or {mask_dir} do not exist!")
        return

    # Preprocess data
    train_generator = ImageMaskGenerator(image_dir=image_dir,
                                     mask_dir=mask_dir,
                                     batch_size=1,
                                     target_size=(256, 256),
                                     shuffle=True)

    # Build and compile the model
    model = unet_model(input_shape=(256, 256, 3), num_classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    model.fit(train_generator, epochs=10)
    model.save(model_dir)

    # Load a test image
    test_image_path = r'C:\Users\singe\Desktop\test_img.png'  # Replace with an actual path
    if os.path.exists(test_image_path):
        test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(256, 256))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255.0
        test_image = tf.expand_dims(test_image, axis=0)  # Add batch dimension

        # Predict mask
        predicted_mask = model.predict(test_image)

        # Visualize results
        plt.subplot(1, 2, 1)
        plt.imshow(test_image[0])  # Original image
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')  # Predicted mask
        plt.title("Predicted Mask")

        plt.show()
    else:
        print(f"Error: Test image not found at {test_image_path}")

# Entry point
if __name__ == "__main__":
    main()