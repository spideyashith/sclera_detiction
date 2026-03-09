import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("sclera_segmentation_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


def predict_mask(image_path):

    image = cv2.imread(image_path)
    orig = image.copy()

    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image = image/255.0
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,0)

    image = torch.tensor(image).float().to(DEVICE)

    with torch.no_grad():
        pred = model(image)

    pred = torch.sigmoid(pred).cpu().numpy()[0][0]

    pred = (pred > 0.5).astype(np.uint8)*255

    pred = cv2.resize(pred,(orig.shape[1],orig.shape[0]))

    return orig, pred

image, mask = predict_mask("images/jaundice/AJ1_20240805_104118.jpg")

# Save mask
cv2.imwrite("predicted_mask.png", mask)

# Extract sclera region
sclera = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite("extracted_sclera.png", sclera)

print("Saved:")
print("predicted_mask.png")
print("extracted_sclera.png")