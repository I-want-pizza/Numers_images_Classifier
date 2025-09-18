import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report


class Trainer:
    """Трейнер для модели: обучение, валидация, тестирование."""

    def __init__(self, model, device, epochs=20, lr=0.001, patience=3, factor=0.5):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience,
                                                              factor=factor)
        self.best_val_accuracy = 0.0
        self.best_model_path = 'best_mnist_model.pth'

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, accuracy, all_predictions, all_targets

    def train_model(self, train_loader, val_loader):
        train_losses = []
        val_accuracies = []
        val_losses = []
        for epoch in range(self.epochs):
            avg_train_loss = self.train_epoch(train_loader)
            avg_val_loss, val_accuracy, all_preds, all_targs = self.validate_epoch(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Сохранена лучшая модель с валидационной точностью: {val_accuracy:.2f}%")

            self.scheduler.step(avg_val_loss)

            print(f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

            if (epoch + 1) % 5 == 0:
                print(f"Отчет по классификации на валидации (эпоха {epoch + 1}):")
                print(classification_report(all_targs, all_preds, digits=4))

        return train_losses, val_losses, val_accuracies

    def evaluate_on_test(self, test_loader):
        _, test_accuracy, all_preds, all_targs = self.validate_epoch(test_loader)  # Переиспользуем validate для теста
        print(f'Точность на тестовом наборе MNIST: {test_accuracy:.2f}%')
        print("\nОтчет по классификации:")
        print(classification_report(all_targs, all_preds, digits=4))
        return test_accuracy
