import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler

from data_loader.data_loader import load_dataset

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
ds_trans = transforms.Compose([transforms.Scale(224),
							   transforms.CenterCrop(224),
							   transforms.ToTensor(),
							   normalize])
# Data loader
train_loader = load_dataset("./data/train")
test_loader = load_dataset("./data/test")

# Loading the model
resnet = models.resnet50(pretrained=True).to(device)

# # Freezing all layers
# for param in resnet.parameters():
# 	param.requires_grad = False

# Freezing the first few layers. Here I am freezing the first 7 layers 
ct = 0
for name, child in resnet.named_children():
	ct += 1
	if ct < 7:
		for name2, params in child.named_parameters():
			params.requires_grad = False
		
resnet.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
# new final layer with 6 classes
num_ftrs = resnet.fc.in_features
print(resnet.fc)
resnet.fc = torch.nn.Linear(num_ftrs, 6).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		print(i)
		
		# Forward pass-
		outputs = resnet(images)
		loss = criterion(outputs, labels)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		scheduler.step()
		
		if (i+1) % 20 == 0:
			print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
resnet.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = resnet(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
