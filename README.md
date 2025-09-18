### Numbers Images Classifier (MNIST, PyTorch)

Классический проект распознавания рукописных цифр MNIST на PyTorch. Репозиторий содержит полный цикл: загрузка и подготовка данных, определение MLP‑модели, обучение с сохранением лучшей версии, оценка на тесте и инференс на своих картинках с визуализацией уверенности модели.

---

### Возможности
- **Обучение**: MLP с `BatchNorm` и `Dropout`, `Adam`, `ReduceLROnPlateau`, сохранение лучшей модели по валидационной точности.
- **Оценка**: точность на тесте и подробный `classification_report`.
- **Инференс на своих изображениях**: чтение через OpenCV, нормализация под статистики MNIST, опция инверсии цвета, визуализация предсказания и уверенности через Matplotlib.
- **Визуализация обучения**: графики loss/accuracy по эпохам.

---

### Стек
- Python 3.10.11
- PyTorch, TorchVision
- OpenCV, NumPy, Matplotlib, scikit‑learn

---

### Структура проекта
```
Numbers_Images_Classifier/
  main.py                # Точка входа: демонстрация загрузки модели и инференса
  models.py              # MLP‑модель для MNIST
  data_handler.py        # Загрузка данных, split train/val/test, DataLoader'ы
  trainer.py             # Цикл обучения/валидации/теста, scheduler, сохранение best модели
  predictor.py           # Инференс на пользовательских изображениях + визуализация
  visualizer.py          # Построение графиков истории обучения
  best_mnist_model.pth   # Готовые веса (если присутствуют)
  requirements.txt       # Зависимости
  data/                  # MNIST будет скачан сюда автоматически
  test_data/             # Примеры изображений для инференса
```

---

### Установка
1) Клонируйте репозиторий и перейдите в папку проекта.
2) Создайте виртуальное окружение и установите зависимости.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Опционально для GPU: установите сборку PyTorch с CUDA согласно официальной инструкции PyTorch.

---

### Быстрый старт (инференс на готовых весах)
В репозитории уже есть файл `best_mnist_model.pth`. Тогда достаточно запустить:

```bash
python main.py
```

Скрипт:
- определит устройство (`cuda` если доступно),
- создаст модель `MNISTModel`,
- загрузит веса из `best_mnist_model.pth`,
- выполнит предсказания для примеров из `test_data/` и отобразит окна с результатами.

---

### Обучение модели с нуля
В `main.py` блок обучения закомментирован — для запуска обучения раскомментируйте соответствующие строки:

```python
from trainer import Trainer
from visualizer import Visualizer

trainer = Trainer(model, device, epochs=30, lr=0.05)
train_losses, val_losses, val_accuracies = trainer.train_model(train_loader, val_loader)
trainer.evaluate_on_test(test_loader)
Visualizer.plot_history(train_losses, val_losses, val_accuracies, "MNIST Training History")
```

Датасет MNIST будет скачан автоматически в папку `data/` при первом запуске. Лучшая модель сохраняется в `best_mnist_model.pth`.

Параметры обучения, которые можно быстро менять:
- **epochs**: число эпох;
- **lr**: скорость обучения для Adam;
- **patience/factor**: для `ReduceLROnPlateau`;
- **batch_size** и **train_val_split**: в `DataHandler`.

---

### Использование предиктора на своих изображениях
Требования к изображению:
- одноцветное (градации серого), но цветные тоже автоматически конвертируются: скрипт читает как grayscale;
- размер в любом случае будет приведён к 28×28, но в таком случае качество может резко упасть;
- нормализация соответствует статистикам MNIST: `mean=0.1307`, `std=0.3081`;
- параметр `invert=True` полезен, если фон белый, а цифра тёмная (как в MNIST).

---

### Детали реализации
- `models.py`: `MNISTModel` — MLP, вход 28×28 → 784, несколько полносвязных слоёв с `BatchNorm1d`, `ReLU`, `Dropout` и финальным `Linear` в 10 классов.
- `trainer.py`: обучение/валидация, `CrossEntropyLoss`, сохранение лучших весов, логирование, отчёты каждые 5 эпох.
- `data_handler.py`: загрузка `torchvision.datasets.MNIST`, преобразования `ToTensor` и `Normalize`, разбиение на train/val.
- `predictor.py`: предобработка через OpenCV, инференс, `softmax` для уверенности, отображение через Matplotlib.
- `visualizer.py`: графики `train_loss`, `val_loss`, `val_accuracy`.



