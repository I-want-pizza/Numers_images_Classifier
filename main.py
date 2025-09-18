import torch

from data_handler import DataHandler
from models import MNISTModel
from predictor import Predictor
from trainer import Trainer
from visualizer import Visualizer


def main():
    print("=" * 80)
    print("ОБУЧЕНИЕ MLP НА MNIST")
    print("=" * 80)

    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Загрузка данных
    data_handler = DataHandler(batch_size=64)
    train_loader, val_loader, test_loader = data_handler.load_datasets_and_loaders()
    if train_loader is None:
        print("Не удалось загрузить данные. Завершение программы.")
        return

    # Создание модели
    model = MNISTModel(hidden_sizes=[512, 256, 128], dropout_rate=0.3)
    print(f"Модель MNIST: {sum(p.numel() for p in model.parameters())} параметров")

    # Загрузка сохранённых весов
    model_path = 'best_mnist_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Веса модели успешно загружены из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке весов модели: {e}")
        print("Убедитесь, что файл 'best_mnist_model.pth' существует и совместим с архитектурой модели")
        return

    # # Обучение
    # trainer = Trainer(model, device, epochs=30, lr=0.05)
    # train_losses, val_losses, val_accuracies = trainer.train_model(train_loader, val_loader)
    #
    # # Тестирование
    # trainer.evaluate_on_test(test_loader)
    #
    # # Визуализация
    # Visualizer.plot_history(train_losses, val_losses, val_accuracies, "MNIST Training History")

    # Предсказание
    predictor = Predictor(model, device)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_0_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_1_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_2_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_3_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_4_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_5_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_6_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_7_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_8_1.png", invert=False)
    predictor.predict_on_image("D:/Geek/Numbers_Images_Classifier/test_data/digit_9 _1.png", invert=False)


if __name__ == "__main__":
    main()
