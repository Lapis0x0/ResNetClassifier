# src/train_model.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

Image.MAX_IMAGE_PIXELS = None

def train_model(data_path, model_save_path, num_epochs=10):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    dataset = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 使用本地的预训练ResNet模型
    model = resnet18()
    state_dict = torch.load('data/models/resnet18-f37072fd.pth')
    model.load_state_dict(state_dict)

    # 获取类别数量并修改最后一层
    num_classes = len(dataset.classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # 训练模型
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
                pbar.update(1)

        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    # 保存模型
    dir_path = os.path.dirname(model_save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(model.state_dict(), model_save_path)

    # 绘制并保存loss值图表
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

    return model, loss_values

if __name__ == "__main__":
    train_model('data/dataset', 'data/models/model.pth')