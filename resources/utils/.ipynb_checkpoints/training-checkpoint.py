import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from sklearn.metrics import f1_score

#---------------------------------- Model ----------------------------------
#-----------------------------------------------------------------------------

def build_model(num_classes=7, backbone='resnet50', pretrained=True):
    if backbone == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise NotImplementedError(f"Unsupported backbone: {backbone}")
    return model

####----------------------------- explainable model ---------------
class Explainable_model(nn.Module):
    def __init__(self, network, num_classes, pretrained=True):
        super(Explainable_model, self).__init__()

        if network == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            backbone = list(model.children())[:-2]
            classifier = nn.Conv2d(model.fc.in_features, num_classes, kernel_size=(1,1), stride=1)
        else:
            raise ValueError(f"Unsupported backbone: {network}. To be implemented")

        self.backbone = nn.Sequential(*backbone)
        self.classifier = classifier

    def forward(self, x):        
        x = self.backbone(x)                # (bs, c, h, w) 
        activation = self.classifier(x)     # (bs, n_class, h, w) 
        bs, c, h, w = x.shape               # (bs, n_class, h, w) 
          
        avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1,1), padding=0)             
        out = avgpool(activation)           # (bs, n_class, 1, 1) 
        out = out.view(out.shape[0], -1)    # (bs, n_class)
        
        return out, activation

####----------------------------- Trained one epoch ---------------
def train_one_epoch(model, loader, criterion, optimizer, device, xai=False):
    model.train()
    model.to(device)
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if xai:
            outputs, acts = model(images)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loss = running_loss / len(loader.dataset)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return loss, all_preds, all_labels

####----------------------------- evaluate after one epoch ---------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, xai=False):
    model.eval()
    model.to(device)
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if xai:
            outputs, acts = model(images)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels