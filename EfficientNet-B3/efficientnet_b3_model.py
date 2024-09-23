import timm
import torch
import torch.nn as nn


class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB3Model, self).__init__()
        self.enetb3 = timm.create_model('efficientnet_b3', pretrained=pretrained)
        self.enetb3.fc = nn.Linear(self.enetb3.fc.in_features, num_classes)

    def forward(self, x):
        x = self.enetb3(x)
        return x

def load_efficientnetb3_model(num_classes, device,task):
    model = EfficientNetB3Model(num_classes)
    model.load_state_dict(torch.load(f'fine_tuned_vit_{task}.pth'))
    model = model.to(device)
    return model
