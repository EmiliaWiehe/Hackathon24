from PIL import Image
import os
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import csv

# Function to randomly overlay images with transformations
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
    shift_x = 0
    for i in range(2):
        temp = random.randint(max_shift_x, 0)
        if temp < shift_x:
            shift_x = temp

    shift_y = 0
    for i in range(2):
        temp = random.randint(max_shift_y, 0)
        if temp < shift_y:
            shift_y = temp

    rotation = 0
    for i in range(2):
        temp = random.randint(0, 360)
        if temp > rotation:
            rotation = temp
    # shift_x = random.randint(max_shift_x, 0)
    # shift_y = random.randint(max_shift_y, 0)
    # rotation = random.randint(0, 360)
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

# class SingleFolderImageDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = folder_path
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png'))]
#         self.label = 2

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.folder_path, img_name)
#         image = Image.open(img_path).convert("RGB")
#         image = transforms.Resize((128, 128))(image)
#         image = transforms.ToTensor()(image)
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, self.label, img_name
    
class OneImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.label = 2

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        image = transforms.Resize((128, 128))(image)
        image = transforms.ToTensor()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.label, self.image_path
    
        
def evaluate_model(model, test_dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, filenames in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)

    return all_predictions, all_labels, all_filenames

# def extract_from_string(s, substring_1, substring_2):
#     start_index = s.find(substring_1)
#     stop_index = s.find(substring_2)
#     if start_index == -1:
#         return None  # Substring not found
#     s = s[start_index:stop_index]
#     #remove non numeric characters
#     s = ''.join(filter(str.isdigit, s))
#     return int(s)

def ML_prediction(part_path, gripper_path, output_path, model):
    # with open(input_csv, "r") as f:
    #     reader = csv.reader(f)
    #     next(reader)  # Skip the header
    #     for row in reader:
    #         part_path = row[0]
    #         gripper_path = row[1]
    temp = False
    while not temp:   
        # Overlay images with transformations
        shift_x, shift_y, rotation, image_path = overlay_images_with_transformations(part_path, gripper_path, output_path)

        # pass combined_image to the model and create a dataset
        # temp_dataset = SingleFolderImageDataset('./result')
        temp_dataset = OneImageDataset(image_path)
        temp_dataloader = DataLoader(temp_dataset, batch_size=1)

        # make predictions
        prediction, labels, filenames = evaluate_model(model, temp_dataloader)
        # print(prediction[0], prediction, labels, filenames)
        # for filename, prediction, label in zip(filenames, predictions, labels):
        if prediction[0] == 1:
            #print(abs(shift_x), abs(shift_y), abs(rotation))
            # write the results to a CSV file
            with open("solution/results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([part_path, gripper_path, abs(shift_x), abs(shift_y), abs(rotation)])
                # empty the output_path folder after writing the results to the CSV file
                # image = Image.open(image_path)
                # image.show()
                # for file in os.listdir(output_path):
                #     file_path = os.path.join(output_path, file)
                #     os.remove(file_path)
            temp = True
            break
                            

        





