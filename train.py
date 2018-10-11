import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 50
learning_rate = 0.001

# Image preprocessing modules
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
ds_trans = transforms.Compose([transforms.resize(224),
							   transforms.CenterCrop(224),
							   transforms.ToTensor(),
							   normalize])

# Data loader
train_loader = load_dataset("./data/train")
test_loader = load_dataset("./data/test")