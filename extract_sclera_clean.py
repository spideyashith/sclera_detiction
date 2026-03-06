import cv2
import numpy as np
import os

INPUT_DIR = "images"
OUTPUT_DIR = "sclera_clean"

os.makedirs(os.path.join(OUTPUT_DIR,"jaundice"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"normal"), exist_ok=True)


def extract_sclera(image_path, save_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    img = cv2.resize(img,(256,256))


    # ---- convert color spaces ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(lab)


    # ---- sclera conditions ----
    bright = v > 160
    low_sat = s < 60
    lab_white = b > 130


    mask = bright & low_sat & lab_white

    mask = mask.astype(np.uint8)*255


    # ---- remove iris ----
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    iris = gray < 70
    mask[iris] = 0


    # ---- morphology ----
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask,5)


    # ---- contours ----
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return False


    contours = sorted(contours,key=cv2.contourArea,reverse=True)

    clean = np.zeros_like(mask)

    for c in contours[:2]:
        cv2.drawContours(clean,[c],-1,255,-1)


    if cv2.countNonZero(clean) < 200:
        return False


    sclera = cv2.bitwise_and(img,img,mask=clean)

    cv2.imwrite(save_path,sclera)

    return True


success = 0
fail = 0


for label in ["jaundice","normal"]:

    folder = os.path.join(INPUT_DIR,label)

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg",".jpeg",".png")):
            continue

        in_path = os.path.join(folder,file)
        out_path = os.path.join(OUTPUT_DIR,label,file)

        ok = extract_sclera(in_path,out_path)

        if ok:
            success+=1
        else:
            fail+=1


print("\nDONE")
print("Success:",success)
print("Failed:",fail)