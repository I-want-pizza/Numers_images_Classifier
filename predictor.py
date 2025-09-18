import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Predictor:
    """Предиктор для кастомных изображений."""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def predict_on_image(self, image_path, invert=True):
        try:
            if not os.path.exists(image_path):
                print(f"Файл не найден: {image_path}")
                return None

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return None

            img = cv2.resize(img, (28, 28))
            if invert:
                img = 255 - img
            img = img.astype('float32') / 255.0
            img = (img - 0.1307) / 0.3081
            img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            print(f"Предсказанная цифра: {predicted_class}")
            print(f"Уверенность: {confidence:.4f}")

            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f'Предсказание: {predicted_class} (уверенность: {confidence:.4f})')
            plt.axis('off')
            plt.show()

            return predicted_class, confidence
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            return None
