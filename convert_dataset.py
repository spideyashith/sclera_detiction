import os
import json
import cv2
import numpy as np

JSON_FOLDER = "sclerasegmentationdataset"
IMAGE_ROOT = "images"

OUT_IMG = "segmentation_dataset/images"
OUT_MASK = "segmentation_dataset/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)


def find_image(filename):
    for root, dirs, files in os.walk(IMAGE_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None


for file in os.listdir(JSON_FOLDER):

    if not file.endswith(".json"):
        continue

    json_path = os.path.join(JSON_FOLDER, file)

    with open(json_path) as f:
        data = json.load(f)

    # IMPORTANT FIX
    image_name = os.path.basename(data["imagePath"])

    image_path = find_image(image_name)

    if image_path is None:
        print("Image not found:", image_name)
        continue

    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:

        if shape["label"] != "sclera":
            continue

        pts = np.array(shape["points"], dtype=np.int32)

        cv2.fillPoly(mask, [pts], 255)

    cv2.imwrite(os.path.join(OUT_IMG, image_name), img)

    mask_name = image_name.replace(".jpg", ".png").replace(".jpeg", ".png")

    cv2.imwrite(os.path.join(OUT_MASK, mask_name), mask)

print("Dataset conversion finished.")