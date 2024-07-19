import csv
import json
import cv2
import numpy as np
import os
import argparse

def create_mask(image_shape, points):
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=True, color=(255), thickness=2)
    cv2.fillPoly(mask, [pts], color=(255))
    return mask

def main(image_dir, annotation_file, mask_dir):
    # Create the mask directory if it does not exist
    os.makedirs(mask_dir, exist_ok=True)

    # Parse the CSV file
    with open(annotation_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename']
            region_shape_attributes = json.loads(row['region_shape_attributes'])
            
            all_points_x = region_shape_attributes['all_points_x']
            all_points_y = region_shape_attributes['all_points_y']
            
            # Load the corresponding image to get its shape
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Image {filename} not found, skipping.")
                continue  # Skip if the image is not found
            
            image_shape = image.shape
            
            # Create mask
            points = list(zip(all_points_x, all_points_y))
            mask = create_mask(image_shape, points)
            
            # Save the mask
            mask_filename = os.path.join(mask_dir, filename)
            cv2.imwrite(mask_filename, mask)

    print(f"Masks have been created and saved in the '{mask_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create masks from annotations.")
    parser.add_argument("--image_dir", default="fun_images", help="Directory containing the original images")
    parser.add_argument("--annotation_file", required=True, help="Path to the CSV annotation file")
    parser.add_argument("--mask_dir", default="masks", help="Directory to save the generated masks")
    
    args = parser.parse_args()
    
    main(args.image_dir, args.annotation_file, args.mask_dir)