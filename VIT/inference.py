# import torch
# from VIT_Model import load_vit_model
# from data import get_data_loaders_cifar10

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     return accuracy

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load CIFAR-10 dataset
#     train_loader, test_loader, num_classes = get_data_loaders_cifar10()
    
#     # Load model
#     model = load_vit_model(num_classes, device)
    
#     # Evaluate model on CIFAR-10 test dataset
#     accuracy = evaluate_model(model, test_loader, device)
#     print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")

# if __name__ == "__main__":
#     main()

# import torch
# from VIT_Model import load_vit_model
# from data import get_data_loaders_cifar10

# def evaluate_model(model, dataloader, device):
#     model.eval()  # Set model to evaluation mode
#     correct, total = 0, 0
#     with torch.no_grad():  # Disable gradient computation
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     return accuracy


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load CIFAR-10 dataset
#     train_loader, test_loader, num_classes = get_data_loaders_cifar10()
    
#     # Load model
#     model = load_vit_model(num_classes, device)
    
#     # Load fine-tuned model weights
#     model.load_state_dict(torch.load('fine_tuned_vit.pth', weights_only=True))
#   # Change to your file name if needed
#     model.eval()  # Set the model to evaluation mode
    
#     # Evaluate model on CIFAR-10 test dataset
#     accuracy = evaluate_model(model, test_loader, device)
#     print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")


# if __name__ == "__main__":
#     main()


import torch
from VIT_Model import load_vit_model 
from data import get_data_loaders_cifar10  

def evaluate_model(model, dataloader, device):
    model.eval()  
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def load_and_evaluate(model_type='fine-tuned'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, test_loader, num_classes = get_data_loaders_cifar10()
    model = load_vit_model(num_classes, device)

    if model_type == 'fine-tuned':
        model.load_state_dict(torch.load('fine_tuned_vit.pth', weights_only=True))
        print("Evaluating fine-tuned model...")
    else:
        print("Evaluating pretrained model...")

    accuracy = evaluate_model(model, test_loader, device)
    print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")


