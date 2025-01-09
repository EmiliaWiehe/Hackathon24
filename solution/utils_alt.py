from PIL import Image
import os
import random
import os
import csv
import math

def calc_gripper_com_offset(gripper_path):
    gripper = Image.open(gripper_path).convert("RGBA")
    # Calculate the size of the square canvas for rotation
    original_width, original_height = gripper.size
    diagonal = int(math.ceil(math.sqrt(original_width**2 + original_height**2)))
    square_size = diagonal  # Ensure square canvas can fully contain rotated image
    
    # Create a square canvas and paste the overlay image at the center
    square_overlay = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 0))
    offset_x = (square_size - original_width) // 2
    offset_y = (square_size - original_height) // 2
    square_overlay.paste(gripper, (offset_x, offset_y))
    
    overlay_image_center_x = square_overlay.width // 2
    overlay_image_center_y = square_overlay.height // 2

    return overlay_image_center_x, overlay_image_center_y

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

# def extract_from_string(s, substring_1, substring_2):
#     start_index = s.find(substring_1)
#     stop_index = s.find(substring_2)
#     if start_index == -1:
#         return None  # Substring not found
#     s = s[start_index:stop_index]
#     #remove non numeric characters
#     s = ''.join(filter(str.isdigit, s))
#     return int(s)

def convert_to_convention(source, x, y, rotation, gripper_path):
    match source:
        case "ANALYTICAL":
            return x, y, (360 - rotation) % 360
        case "COMPACT":
            # Offset shift by the center of the gripper image
            gripper_x_offset, gripper_y_offset = calc_gripper_com_offset(gripper_path)
            return abs(x)+gripper_x_offset, abs(y)+gripper_y_offset, (360 - rotation) % 360
        
def place_in_center(part_path, gripper_path):
    part = Image.open(part_path).convert("RGBA")
    gripper = Image.open(gripper_path).convert("RGBA")
    part_width, part_height = part.size
    gripper_width, gripper_height = gripper.size
    angle = 0

    if gripper_width > gripper_height:
        angle = 90
        gripper = gripper.rotate(90, expand=True)
        gripper_width, gripper_height = gripper.size

    # Place center of gripper in part center
    x = (part_width - gripper_width) // 2 + gripper_width // 2
    y = (part_height - gripper_height) // 2 + gripper_height // 2

    return x, y, angle

def ML_prediction(part_path, gripper_path, output_path, model):
    # with open(input_csv, "r") as f:
    #     reader = csv.reader(f)
    #     next(reader)  # Skip the header
    #     for row in reader:
    #         part_path = row[0]
    #         gripper_path = row[1]
    temp = False
    found_result = False
    
    # If no result is found, place the gripper in the center of the part
    if not found_result:
        position = place_in_center(part_path, gripper_path)
        with open(output_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([part_path, gripper_path, position[0], position[1], position[2]])
                            

        





