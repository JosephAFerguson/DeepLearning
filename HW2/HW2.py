# Joe Ferguson Deep Learning HW2 - PyTorch version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Num GPUs Available:", torch.cuda.device_count())

transform = transforms.Compose([
    transforms.ToTensor(),                  
    transforms.Normalize((0.0,), (1.0,)) 
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

val_size = 50
test_size = 50
test_dataset, val_dataset = random_split(test_dataset, [len(test_dataset) - val_size - test_size, val_size + test_size])
val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

n = 50
fig, axes = plt.subplots(math.ceil(n/10), 10, figsize=(20,10))
axes = axes.flatten()
for i in range(n):
    img, label = train_dataset[i]
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"Label - {label}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

DNNmodel = SimpleNN().to(device)

class CNN(nn.Module):
   def __init__(self):
       super(CNN, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
       self.fc1 = nn.Linear(1568, 128) 
       self.fc2 = nn.Linear(128, 10) 
   def forward(self, x):
       x = self.pool(F.relu(self.conv1(x)))
       x = self.pool(F.relu(self.conv2(x)))
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return x
   
CNNmodel = CNN().to(device)

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)   
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

   
CNN2model = CNN_2().to(device)

ResNetmodel = models.resnet18(weights=None)
ResNetmodel.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)
ResNetmodel.fc = nn.Linear(ResNetmodel.fc.in_features, 10)
ResNetmodel = ResNetmodel.to(device)

def train_model(model, epochs=10, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%")
    return model

def test_model(model, epochs=10, lr=0.01):
    model = train_model(model,epochs=epochs, lr=lr)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

if __name__ == "__main__":
    test_model(CNN2model, 10, 0.001)