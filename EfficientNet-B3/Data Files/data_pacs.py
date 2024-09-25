import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class PACS(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.categories = os.listdir(os.path.join(root_dir, domain))
        self.images = []
        self.labels = []

        for category in self.categories:
            category_dir = os.path.join(root_dir, domain, category)
            for image_file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, image_file)
                self.images.append(image_path)
                self.labels.append(self.categories.index(category))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders_pacs(batch_size=64):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load SVHN dataset
    train_dataset = PACS(root='./data', domain='photo',transform=transform)
    test_dataset_art = PACS(root='./data', domain='art_painting', transform=transform)
    test_dataset_cartoon= PACS(root='./data', domain='cartoon', transform=transform)
    test_dataset_sketches=PACS(root='./data', domain='sketches', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_art = DataLoader(test_dataset_art, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_cartoon = DataLoader(test_dataset_cartoon, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_sketches = DataLoader(test_dataset_sketches, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader_art, test_loader_cartoon, test_loader_sketches, 7 
        
