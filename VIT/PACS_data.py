import deeplake
from torch.utils.data import DataLoader
from torchvision import transforms

# Load PACS dataset from Deep Lake
ds = deeplake.load("hub://activeloop/pacs-train")

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset Class to wrap Deep Lake dataset
class DeepLakePACS(torch.utils.data.Dataset):
    def __init__(self, deeplake_dataset, transform=None):
        self.ds = deeplake_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds['image'])

    def __getitem__(self, idx):
        image = self.ds['image'][idx].numpy()
        label = int(self.ds['label'][idx].numpy())
/kaggle/working/fine_tuned_vit.pth
        if self.transform:
            image = self.transform(image)

        return image, label

# Create DataLoader for PACS
pacs_dataset = DeepLakePACS(ds, transform=transform)
pacs_loader = DataLoader(pacs_dataset, batch_size=64, shuffle=True)

# Now pacs_loader can be used for training or evaluation

