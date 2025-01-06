import csv
import os
import sys
import random
from utils import ML_prediction
from model import SimpleCNN
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def main(input_csv, output_path):

    # create results.csv file with headers
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "gripper", "x", "y", "rotation"])
        print("results.csv created successfully")

    # load ML model
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded successfully from {model_path} and set to evaluation mode") 

    # Read the input CSV file
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        # Loop through each row in the CSV file
        for row in reader:
            part_path = row[0]
            gripper_path = row[1]
            print(f"Part path: {part_path}, Gripper path: {gripper_path}")


            # Max code
            # analytic_result = max_function(image, output_path)

            analytic_result = False  # Set to True if we don't have to continue with the ML model

            # This function calls the Machine Learning model to predict the x, y, and rotation values
            # The results are printed to the console and written to "results.csv"
            if analytic_result == False:
                ML_prediction(part_path, gripper_path, output_path, model)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py path/to/input/tasks.csv path/to/output/folder", file=sys.stderr)
        sys.exit(1)
    
    # Get the input CSV file and output folder from the command line arguments
    input_csv = sys.argv[1]
    output_path = sys.argv[2]
    
    main(input_csv, output_path)