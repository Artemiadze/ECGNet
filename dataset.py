import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ECGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.image_files = []
        self.labels = []

        print(f"Found {len(self.files)} files in {root_dir}")

        class_map = {
            "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 1,
            "Normal Person ECG Images (284x12=3408)": 0
        }

        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for file in os.listdir(class_path):
                    if file.endswith((".jpg", ".png", ".jpeg")):
                        self.image_files.append(os.path.join(class_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_dir, test_dir, batch_size=32, img_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = ECGDataset(train_dir, transform)
    test_dataset = ECGDataset(test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

