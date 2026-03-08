import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20


class ScleraDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg",".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        transformed = self.transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"]

        image = image / 255.0
        mask = mask / 255.0

        image = np.transpose(image, (2,0,1))

        return torch.tensor(image).float(), torch.tensor(mask).unsqueeze(0).float()


dataset = ScleraDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss:", total_loss)

torch.save(model.state_dict(), "sclera_segmentation_model.pth")
print("Model saved!")