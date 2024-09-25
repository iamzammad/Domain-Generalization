import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader

def get_data_loaders_svhn(batch_size=64):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load SVHN dataset
    train_dataset = SVHN(root='./data', train=True, download=True, transform=transform)
    test_dataset = SVHN(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, 10 
