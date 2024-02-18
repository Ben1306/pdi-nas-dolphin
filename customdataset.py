import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Get the list of class directories
        self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # List to store image paths and corresponding labels
        self.samples = []

        # Load image paths and labels
        for i, cls in enumerate(self.classes):
            class_dir = os.path.join(root, cls)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpeg', '.JPEG', '.jpg', '.png')):
                    path = os.path.join(class_dir, filename)
                    label = i
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform is not None:
            img = self.transform(img)

        return img, target