import torchvision
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms  # Nueva versión de transforms
import matplotlib.pyplot as plt

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 10
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST("./data/", train=True, download=True, transform=transform)
valid_dataset = torchvision.datasets.MNIST("./data/", train=False, download=True, transform=transform)


train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, BATCH_SIZE)

train_N = len(train_dataset)
valid_N = len(valid_dataset)

class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)

flattened_img_size = 75 * 3 * 3

# Input 1 x 28 x 28
base_model = nn.Sequential(
    MyConvBlock(IMG_CHS, 25, 0), # 25 x 14 x 14
    MyConvBlock(25, 50, 0.2), # 50 x 7 x 7
    MyConvBlock(50, 75, 0),  # 75 x 3 x 3
    # Flatten to Dense Layers
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, N_CLASSES)
)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(base_model.parameters())

model = base_model.to(device)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train():
    total_loss = 0
    total_correct = 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / train_N
    print(f'Train - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

def validate():
    total_loss = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_function(output, y)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(valid_loader)
    avg_acc = total_correct / valid_N
    print(f'Valid - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

import torch._dynamo
torch._dynamo.config.suppress_errors = True

epochs = 6

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

"""# ✅ Alta precisión (accuracy)
en validación, ya que refleja cuántas predicciones fueron correctas.
# 📉 Baja pérdida (loss)
en validación, ya que indica cuán confiables son las predicciones (más bajo es mejor).

| Época | Valid Accuracy | Valid Loss |
| ----- | -------------- | ---------- |
| 0     | 0.9851         | 0.0452     |
| 1     | 0.9918         | 0.0270     |
| 2     | 0.9908         | 0.0252     |
| 3     | 0.9919         | 0.0223     |
| 4     | 0.9932 ✅       | 0.0220 ✅   |
| 5     | 0.9927         | 0.0237     |

Otra forma de determinar la mejor época es una balanceo entre la precision y pérdida: Score=Validation Accuracy−λ×Validation Loss

| Época | Accuracy | Loss     | Score                      |
| ----- | -------- | -------- | -------------------------- |
| 0     | 0.9851   | 0.0452   | 0.9851 - 0.0452 = 0.9399   |
| 1     | 0.9918   | 0.0270   | 0.9918 - 0.0270 = 0.9648   |
| 2     | 0.9908   | 0.0252   | 0.9908 - 0.0252 = 0.9656   |
| 3     | 0.9919   | 0.0223   | 0.9919 - 0.0223 = 0.9696   |
| 4     | 0.9932 ✅ | 0.0220 ✅ | 0.9932 - 0.0220 = 0.9712 ✅ |
| 5     | 0.9927   | 0.0237   | 0.9927 - 0.0237 = 0.9690   |
"""

torch.save(base_model, 'model.pth')
