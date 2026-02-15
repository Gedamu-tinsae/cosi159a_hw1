import torch
import torchvision.models as models

# Call the ResNet-50 model from torchvision

# resnet18 vs resnet50
model = models.resnet18(weights='IMAGENET1K_V1')

# Print the network to see the detailed architecture
print(model)