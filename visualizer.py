import matplotlib.pyplot as plt


class Visualizer:
    """Визуализатор истории обучения."""

    @staticmethod
    def plot_history(train_losses, val_losses, val_accuracies, title="Training History"):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(val_losses)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        ax3.plot(val_accuracies)
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
