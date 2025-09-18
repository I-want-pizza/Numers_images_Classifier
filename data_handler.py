from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


class DataHandler:
    """Обработчик данных для MNIST: загрузка, разделение и создание лоадеров."""

    def __init__(self, data_dir='./data', batch_size=64, train_val_split=0.8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_datasets_and_loaders(self):
        try:
            full_train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

            train_size = int(self.train_val_split * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return train_loader, val_loader, test_loader
        except Exception as e:
            print(f"Ошибка при загрузке данных MNIST: {e}")
            return None, None, None
