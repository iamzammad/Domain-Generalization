import torch
from efficientnet_b3_model import load_efficientnetb3_model
from data_cifar10 import get_data_loaders_cifar10
from sklearn.metrics import confusion_matrix

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

    conf_matrix=confusion_matrix(labels,predicted)
    classwise_accuracies=np.zeros((10,1))
    for i in range(10):
        total_class_labels=0
        for j in range(10):
            total_class_labels += conf_matrix[i,j]
        classwise_accuracies[i,0]=conf_matrix[i,i]/total_class_labels
    accuracy = 100 * correct / total

    print("Confusion Matrix")
    print(conf_matrix)
    print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")
    print("Classwise Accuracies:")
    print(classwise_accuracies}
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, num_classes = get_data_loaders_cifar10()
    
    # Load model
    model = load_efficientnetb3_model(num_classes, device,task='task2')
    
    evaluate_model(model, test_loader, device)
    

if __name__ == "__main__":
    main()
