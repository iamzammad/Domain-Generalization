import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
from VIT_Model import load_vit_model
from data import get_data_loaders_cifar10

# Fine-tuning Function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, num_classes = get_data_loaders_cifar10()
    
    # Load model
    model = load_vit_model(num_classes, device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)
    
    # Optionally, you could save the model after training
    torch.save(model.state_dict(), 'fine_tuned_vit.pth')

if __name__ == "__main__":
    main()
