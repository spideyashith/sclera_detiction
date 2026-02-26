import os
import cv2
import albumentations as A

input_dir = "images"        # Original images
output_dir = "images_aug"  # Augmented output

os.makedirs(output_dir, exist_ok=True)

augment = A.Compose([

    A.Rotate(limit=5, p=0.5),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.GaussianBlur(blur_limit=3, p=0.3),

    A.RandomScale(scale_limit=0.05, p=0.4)

])

count = 0

for file in os.listdir(input_dir):

    if not file.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, file)
    img = cv2.imread(path)

    if img is None:
        continue

    for i in range(2):   # 2 new images per image

        augmented = augment(image=img)["image"]

        name = file.replace(".jpg", f"_aug{i}.jpg")
        out_path = os.path.join(output_dir, name)

        cv2.imwrite(out_path, augmented)

        count += 1

print("Augmentation complete!")
print("Generated images:", count)
