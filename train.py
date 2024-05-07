import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 用于创建MLP和KAN的通用模块
from KAN import KANLinear  # 假设已经实现并放在src/efficient_kan.py文件中

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 适应ResNet的输入大小
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# define MLP
class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 512)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, out_features)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# ResNet with MLP/KAN
class ResNetWithCustomLayer(torch.nn.Module):
    def __init__(self, custom_layer):
        super(ResNetWithCustomLayer, self).__init__()
        base_model = resnet18(weights=False)
        self.features = torch.nn.Sequential(*list(base_model.children())[:-1])  # remove last classification layer
        self.custom_layer = custom_layer(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.custom_layer(x)
        return x


models = {
    'KAN': ResNetWithCustomLayer(KANLinear),
    'MLP': ResNetWithCustomLayer(MLP)
}


def train_and_test(model, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {'train_loss': [], 'test_accuracy': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_accuracy = 100. * correct / len(test_loader.dataset)
        history['test_accuracy'].append(test_accuracy)
        print(f'Epoch: {epoch} - Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    return history


# 训练并绘制结果
results = {}
for key, model in models.items():
    print(f'Training {key} model...')
    model.to(device)
    model.cuda()  # 训练前确保模型在GPU上
    results[key] = train_and_test(model)
    print(f'{key} model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# 绘制训练损失和测试准确率
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for key, result in results.items():
    axs[0].plot(result['train_loss'], label=f'{key} Train Loss')
    axs[1].plot(result['test_accuracy'], label=f'{key} Test Accuracy')

axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].set_title('Test Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()

plt.show()
