import timm
import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, finetune="classifier"):
        super(ViTClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        if finetune == "classifier":
            #freezing the backbone
            for param in self.vit.parameters():
                param.requires_grad = False
            #unfreezing the classifier
            for param in self.vit.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vit(x)

def load_vit_model(num_classes, device):
    model = ViTClassifier(num_classes)
    model = model.to(device)
    return model



# state = model.state_dict()
# for k, v in state.items():
#     print(k)