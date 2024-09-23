import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
from efficientnet_b3_model import load_efficientnetb3_model
from data_cifar10 import get_data_loaders_cifar10

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    scaler = GradScaler()  
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time() 
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(): 
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - start_time:.2f} seconds. "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, num_classes = get_data_loaders_cifar10()
    model = load_efficientnetb3_model(num_classes, device, task='nopath')

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    for name, param in model.named_parameters():
        if "some_specific_layer" in name: 
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)

    torch.save(model.state_dict(), 'fine_tuned_enetb3_task2.pth')


if __name__ == "__main__":
    main()
