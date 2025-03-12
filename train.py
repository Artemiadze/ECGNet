import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders  # Функция для загрузки датасета

# Добавляем папку с моделью в PYTHONPATH
sys.path.append("model")
from model.CNN import buildModel_2DCNN  # Импортируем функцию для создания модели

# Пути к данным
train_dir = "dataset/train"
test_dir = "dataset/test"

"""# Define paths to the dataset directories
data_dir = '/kaggle/input/ecg-analysis/ECG_DATA'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')"""

# Получаем DataLoader'ы
train_loader, test_loader = get_dataloaders(train_dir, test_dir)

"""# Инициализация модели
model = buildModel_2DCNN(1, last_layer='linear')  # 1 канал (Ч/Б изображения)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)"""

# Инициализация модели
model = buildModel_2DCNN(1, last_layer='linear')
model.summary()  # 1 канал (Ч/Б изображения)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {total_loss:.4f}")

# Сохранение модели
torch.save(model.state_dict(), "cnn_ecg_model.pth")
print("Модель сохранена!")


