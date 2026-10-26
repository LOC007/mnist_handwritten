import torch
from torch.utils.data import TensorDataset, DataLoader

from data_parser import load_images, load_labels
from model import MnistNet


train_x = load_images(r"D:\MNIST_Task\data\train-images-idx3-ubyte")
train_y = load_labels(r"D:\MNIST_Task\data\train-labels-idx1-ubyte")
test_x  = load_images(r"D:\MNIST_Task\data\t10k-images-idx3-ubyte")
test_y  = load_labels(r"D:\MNIST_Task\data\t10k-labels-idx1-ubyte")

train_ds = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MnistNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out = model(xb)
        loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"第 {epoch+1} 轮, 最后一批 loss = {loss.item():.4f}")
test_ds = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

print(f"测试集准确率: {correct/total*100:.2f}%")

torch.save(model.state_dict(), "model.pth")
print("模型已保存到 model.pth")
