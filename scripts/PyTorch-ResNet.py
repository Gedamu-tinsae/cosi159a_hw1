import torch
import torchvision.models as models

# Call the ResNet-50 model from torchvision
model = models.resnet50(weights='IMAGENET1K_V1')

# Print the network to see the detailed architecture
print(model)