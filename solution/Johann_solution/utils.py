from PIL import Image
import os
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Function to overlay images with transformations
def overlay_images_with_transformations(part_path, gripper_path, output_path):

    # Load images
    part = Image.open(part_path).convert("RGBA")
    gripper = Image.open(gripper_path).convert("RGBA")

    # check if the gripper image is larger than the part image
    if part.width < gripper.width or part.height < gripper.height:
        print("Gripper image is larger than the part image. Exiting...")
        return -1, -1, -1
    
    # the shift_x and shift_y are the values that are used to shift the gripper image
    # the shift value is calculated by placing the gripper image on the top left corner of the part image
    # and then shifting it by the shift value

    # (0, 0) is the top left corner of the image
    start_x = 0 
    start_y = 0
    max_shift_x = -(part.width - gripper.width) // 2  # negative value because we want to shift the gripper to the right
    max_shift_y = -(part.height - gripper.height) // 2 # negative value because we want to shift the gripper down
   

    # Random shift values should be negative only and should be less than part width - gripper width
    shift_x = random.randint(max_shift_x, 0)
    shift_y = random.randint(max_shift_y, 0)
    rotation = random.randint(0, 360)
    shift_x = start_x + max(min(shift_x, -max_shift_x), max_shift_x)
    shift_y = start_y + max(min(shift_y, -max_shift_y), max_shift_y)
    # Transformations to apply
    transformations = [
        {'type': 'rotate', 'value': rotation},
        {'type': 'shift', 'x': shift_x, 'y': shift_y}
    ]

    # Note: We don't resize the images now, but during training we will resize the images to a fixed size
    
    # Apply rotation transformation
    for transform in transformations:
        if transform['type'] == 'rotate':
            gripper = gripper.rotate(transform['value'], expand=True)
    
    # Apply shift transformation
    for transform in transformations:
        if transform['type'] == 'shift':
            shift_x = start_x + max(min(transform['x'], -max_shift_x), max_shift_x) 
            shift_y = start_y + max(min(transform['y'], -max_shift_y), max_shift_y) 
            gripper = gripper.transform(part.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
            # print("Shift x: ", shift_x, "Shift y: ", shift_y, "Rotation: ", rotation)
    
    # store the shift values and rotation as string in the output path
    output_path = output_path + "/_x_" + str(shift_x) + "_y_" + str(shift_y) + "_rotation_" + str(rotation) + ".png"

    # Overlay images
    combined = Image.alpha_composite(part, gripper)
    combined.save(output_path)

    return shift_x, shift_y, rotation, output_path


def prediction(model, dataloader):
    model.eval()
    # files = []
    predictions = []
    for images, file_names in dataloader:
        outputs = model(images)
        #print(outputs)
        _, predicted = torch.max(outputs, 1)
        # files.append(file_names)
        predictions.append(predicted)
    return predictions #, files

class SingleFolderImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        # convert the image to tensor
        # resizes the image to 128x128
        image = transforms.Resize((128, 128))(image)
        image = transforms.ToTensor()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name
    

def get_dataloader(folder_path, batch_size=32, shuffle=True, transform=None):
    dataset = SingleFolderImageDataset(folder_path, transform=transform)
    print("Dataset created successfully")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def check_tuple_positions(tuples_list):
    for idx, tpl in enumerate(tuples_list):
        if tpl[1] == 1:
            return idx
        





