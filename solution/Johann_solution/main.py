import csv
import os
import sys
import random
from utils import overlay_images_with_transformations, prediction, SingleFolderImageDataset
from utils import check_tuple_positions, get_dataloader
from model import SimpleCNN
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def main(input_csv, output_path):

    # load ML model
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    print("Model path: ", model_path)

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully") 

    # Read the input CSV file
    #Firstly, it receives a CSV file containing the part and gripper image file names
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            part_path = row[0]
            gripper_path = row[1]
            print(f"Part path: {part_path}, Gripper path: {gripper_path}")
            for i in range(32):
                # Overlay images with transformations
                shift_x, shift_y, rotation, image_path = overlay_images_with_transformations(part_path, gripper_path, output_path)
                print("Images overlayed successfully")
                # pass combined_image to the model
                # create a dataset
                dataset = SingleFolderImageDataset('./result')
                dataloader = DataLoader(dataset, batch_size=1)
                print("Dataset and DataLoader created successfully")
                # make predictions
                predictions = prediction(model, dataloader)
                print(predictions)
                break

            # print("Images overlayed successfully")

            # # get DataLoader
            # dataloader = get_dataloader(output_path, batch_size=32)
            # print("Dataloader created successfully")

            # # make predictions
            # predictions, files = prediction(model, dataloader)
            # print(predictions)
            # idx = check_tuple_positions(predictions)
            #print(f"Predicted class: {predictions[idx][1]}, File name: {predictions[idx][0]}")
      

    # create dataset and dataloader



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py path/to/input/tasks.csv path/to/output/folder", file=sys.stderr)
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_path = sys.argv[2]
    
    main(input_csv, output_path)