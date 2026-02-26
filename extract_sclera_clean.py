import cv2
import numpy as np
import os


# -----------------------------
# FOLDERS
# -----------------------------
INPUT_DIR = "images_aug"
OUTPUT_DIR = "sclera_clean_aug"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def extract_sclera(image_path, save_path):

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Cannot read:", image_path)
        return False


    # Resize for consistency
    img = cv2.resize(img, (256, 256))


    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # -----------------------------
    # SCLERA COLOR RANGE (TUNED)
    # -----------------------------
    lower = np.array([0, 10, 160])
    upper = np.array([40, 120, 255])

    mask = cv2.inRange(hsv, lower, upper)


    # -----------------------------
    # MORPHOLOGY CLEANING
    # -----------------------------
    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)


    # -----------------------------
    # KEEP ONLY BIGGEST REGION
    # -----------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("⚠️ No contour:", image_path)
        return False


    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)


    # -----------------------------
    # REMOVE SMALL AREAS
    # -----------------------------
    area = cv2.countNonZero(clean_mask)

    if area < 500:
        print("⚠️ Too small area:", image_path)
        return False


    # -----------------------------
    # APPLY MASK
    # -----------------------------
    sclera = cv2.bitwise_and(img, img, mask=clean_mask)


    # -----------------------------
    # SAVE
    # -----------------------------
    cv2.imwrite(save_path, sclera)

    return True



# -----------------------------
# PROCESS ALL IMAGES
# -----------------------------
success = 0
fail = 0


for file in os.listdir(INPUT_DIR):

    if not file.lower().endswith(".jpg"):
        continue


    in_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUTPUT_DIR, file)

    ok = extract_sclera(in_path, out_path)

    if ok:
        success += 1
    else:
        fail += 1


print("\n===== DONE =====")
print("Success:", success)
print("Failed :", fail)
