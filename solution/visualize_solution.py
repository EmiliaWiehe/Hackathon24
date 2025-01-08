import csv
import os
from PIL import Image
from matplotlib import pyplot as plt
import math

def overlay_images(base_path, overlay_path, x_offset, y_offset, angle):
    """
    Overlays one image onto another with specified offsets and rotation,
    making all non-transparent pixels in the overlay image fully opaque.

    Parameters:
        base_path (str): Path to the base .png file.
        overlay_path (str): Path to the overlay .png file.
        x_offset (int): X offset of the overlay image.
        y_offset (int): Y offset of the overlay image.
        angle (float): Angle in degrees to rotate the overlay image.

    Returns:
        Image: The resulting image with the overlay applied.
    """

    angle = -angle  # Reverse the angle to match the orientation of the overlay image

    # Open the base and overlay images
    base_image = Image.open(base_path).convert("RGBA")
    overlay_image = Image.open(overlay_path).convert("RGBA")
    
    # Process the overlay image to make all non-transparent pixels fully opaque
    data = overlay_image.getdata()
    new_data = []
    for item in data:
        # If the alpha channel is greater than 0, set it to 255 (fully opaque)
        if item[3] > 0:
            new_data.append((item[0], item[1], item[2], 200))
        else:
            new_data.append(item)
    overlay_image.putdata(new_data)

    # Calculate the size of the square canvas for rotation
    original_width, original_height = overlay_image.size
    diagonal = int(math.ceil(math.sqrt(original_width**2 + original_height**2)))
    square_size = diagonal  # Ensure square canvas can fully contain rotated image
    
    # Create a square canvas and paste the overlay image at the center
    square_overlay = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 0))
    offset_x = (square_size - original_width) // 2
    offset_y = (square_size - original_height) // 2
    square_overlay.paste(overlay_image, (offset_x, offset_y))

    # Rotate the square overlay image
    overlay_image_rotated = square_overlay.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    overlay_image_center_x = overlay_image_rotated.width // 2
    overlay_image_center_y = overlay_image_rotated.height // 2
    
    # Create a transparent image the same size as the base image
    overlay_canvas = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    
    # Paste the rotated overlay image onto the transparent canvas at the specified position
    overlay_canvas.paste(overlay_image_rotated, (x_offset-overlay_image_center_x, 
        y_offset-overlay_image_center_y), overlay_image_rotated)
    
    # Composite the base image and the overlay canvas
    result = Image.alpha_composite(base_image, overlay_canvas)
    
    return result


def extract_png_name(filepath):
    # Ensure the file has a .png extension
    if filepath.lower().endswith('.png'):
        # Extract the file name without the extension
        return os.path.splitext(os.path.basename(filepath))[0]
    else:
        return None


def main():
    # Delete all images in the visualization folder
    for file in os.listdir("solution/visualization"):
        if file.endswith(".png"):
            os.remove(os.path.join("solution/visualization", file))

    # Import result.csv file
    result_csv = "solution/results.csv"

    # Read the input CSV file
    with open(result_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        # Loop through each row in the CSV file
        for row in reader:
            if len(row) < 5:
                continue  # Skip rows that do not have enough columns
            part_path = row[0]
            gripper_path = row[1]
            x = row[2]
            y = row[3]
            rotation = row[4]

            image = overlay_images(part_path, gripper_path, int(x), int(y), int(rotation))
            
            #image = overlay_images(part_path, gripper_path, 0, 0, 0)
            part_name = extract_png_name(part_path)
            gripper_name = extract_png_name(gripper_path)
            combined_name = f"{part_name}_{gripper_name}_overlay.png"
            # Save the combined image
            image.save(f"solution/visualization/{combined_name}")

            print(f"Part path: {part_path}, Gripper path: {gripper_path}, x: {x}, y: {y}, rotation: {rotation}")


if __name__ == "__main__":
    main()