import sys
import torch
from dataset import get_dataloaders  # Функция для загрузки датасета

# Добавляем папку с моделью в PYTHONPATH
sys.path.append("model")
from model.CNN import buildModel_2DCNN  # Импортируем функцию для создания модели

# Пути к данным
test_dir = "dataset/test"
"""# Define paths to the dataset directories
data_dir = '/kaggle/input/ecg-analysis/ECG_DATA'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')"""

# Получаем DataLoader
_, test_loader = get_dataloaders("dataset/train", test_dir)

# Загружаем модель
model = buildModel_2DCNN(1, last_layer='linear')
model.load_state_dict(torch.load("cnn_ecg_model.pth"))
model.eval()

# Тестирование модели
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Точность модели: {accuracy:.2f}%")
