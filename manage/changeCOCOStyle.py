import json

def convert_to_coco_format(json_data):
    coco_data = {
        "info": {
            "description": "Annotation in COCO format",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "Your Name",
            "date_created": "2024-05-05"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id = json_data["Image_id"]
    image_filename = json_data["Image_filename"]
    image_width = json_data["Image_width"]
    image_height = json_data["Image_height"]

    # Add image information to COCO data
    coco_data["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": image_width,
        "height": image_height
    })

    # Iterate over text coordinates and annotations
    for text_coord in json_data["Text_Coord"]:
        bbox = text_coord["Bbox"]
        annotate = text_coord["annotate"]

        # Add annotation information to COCO data
        coco_data["annotations"].append({
            "id": len(coco_data["annotations"]) + 1,
            "image_id": image_id,
            "category_id": 1,  # Assuming only one category for simplicity
            "bbox": bbox,
            "area": bbox[2] * bbox[3],  # Calculating area
            "iscrowd": 0,
            "segmentation": [],
            "ignore": 0
        })

    return coco_data

def save_coco_format(coco_data, output_file):
    with open(output_file, "w") as f:
        json.dump(coco_data, f)

# Load JSON data
with open("your_annotation.json", "r") as f:
    json_data = json.load(f)

# Convert to COCO format
coco_data = convert_to_coco_format(json_data)

# Save COCO format to file
save_coco_format(coco_data, "output_coco.json")
