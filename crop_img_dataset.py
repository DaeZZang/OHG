import json
import cv2
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# Function to read JSON file and extract data
def read_json_label(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to process each image
def process_image(file_info):
    filename, dataset_dir, labels_dir, save_dir = file_info
    if filename.endswith('.json'):
        base_filename = filename.replace('.json', '')
        image_filename = base_filename + '.png'
        image_path = os.path.join(dataset_dir, image_filename)
        label_path = os.path.join(labels_dir, filename)
        label_data = read_json_label(label_path)
        crop_and_save_image(image_path, label_data, save_dir)

# Function to crop and save images based on bbox
def crop_and_save_image(image_path, label_data, save_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]

    for char in label_data['Text_Coord']:
        x, y, w, h = char['Bbox'][:4]
        # Adjust coordinates and dimensions to ensure they are within image boundaries
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_width - x), min(h, img_height - y)

        if w <= 0 or h <= 0:
            print(f"Invalid Bbox for {image_path}: ({x}, {y}, {w}, {h}) - skipping")
            continue

        cropped_img = img[y:y+h, x:x+w]
        
        if cropped_img.size == 0:
            print(f"Empty crop for {image_path}: ({x}, {y}, {w}, {h}) - file will be skipped")
            continue

        char_folder = os.path.join(save_dir, char['annotate'])
        os.makedirs(char_folder, exist_ok=True)
        filename = f"{Path(image_path).stem}_{x}_{y}_{w}_{h}.png"
        cv2.imwrite(os.path.join(char_folder, filename), cropped_img)

# Define directories
dataset_dir = './dataset/train/images2'
labels_dir = './dataset/train/labels'
save_dir = './dataset/train/crop'

# Prepare multiprocessing with tqdm for progress monitoring
file_list = [(filename, dataset_dir, labels_dir, save_dir) for filename in os.listdir(labels_dir)]

# Create a Pool and map the process_image function
with Pool(processes=8) as pool:
    r = list(tqdm(pool.imap(process_image, file_list), total=len(file_list)))

print("Processing completed.")
