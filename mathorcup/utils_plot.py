import matplotlib.pyplot as plt
import numpy as np
import random


class Myplot:
    def __init__(self, row, col, figsize=(10, 6), dpi=200) -> None:
        plt.rcParams.update({'font.size': 6})
        self.fig, self.axes = plt.subplots(
            nrows=row, ncols=col, figsize=figsize, dpi=dpi
        )
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def save(self, path):
        plt.savefig(path)

    def fresh(self):
        # 重新绘制图表
        plt.draw()
        plt.pause(0.1)  # 暂停一段时间，使图表得以更新

    def generate_random_list(self, y, x, min_val, max_val):
        random_list = []
        for _ in range(6):
            random_element = random.uniform(min_val, max_val)
            random_list.append(random_element)
        return random_list

    def plot_accuracy_curve(
        self,
        y,
        x,
        epochs,
        train_accuracy,
        val1_accuracy=None,
        val2_accuracy=None,
        val3_accuracy=None,
    ):
        self.axes[y, x].clear()
        self.axes[y, x].plot(epochs, train_accuracy, label="Training Accuracy")
        if val1_accuracy is not None:
            self.axes[y, x].plot(epochs, val1_accuracy, label="Validation Accuracy")
        if val2_accuracy is not None:
            self.axes[y, x].plot(epochs, val2_accuracy, label="ValAcc_Sigmoid")
        if val3_accuracy is not None:
            self.axes[y, x].plot(epochs, val3_accuracy, label="ValAcc_Tanh")
        self.axes[y, x].set_title("Accuracy Curve")
        self.axes[y, x].set_xlabel("Epochs")
        self.axes[y, x].set_ylabel("Accuracy")
        self.axes[y, x].legend()
        self.axes[y, x].set_ylim([0, 1])

    def plot_learning_curve(self, y, x, epochs, train_loss, val_loss=None):
        self.axes[y, x].clear()
        self.axes[y, x].plot(epochs, train_loss, label="Training Loss")
        if val_loss is not None:
            self.axes[y, x].plot(epochs, val_loss, label="Validation Loss")
        self.axes[y, x].set_title("Learning Curve")
        self.axes[y, x].set_xlabel("Epochs")
        self.axes[y, x].set_ylabel("Loss")
        self.axes[y, x].legend()

    def plot_roc_curve(self, y, x, fpr, tpr):
        self.axes[y, x].clear()
        self.axes[y, x].plot(fpr, tpr)
        self.axes[y, x].set_title("ROC Curve")
        self.axes[y, x].set_xlabel("False Positive Rate")
        self.axes[y, x].set_ylabel("True Positive Rate")
        self.axes[y, x].set_xlim([0, 1])
        self.axes[y, x].set_ylim([0, 1])

    def plot_confusion_matrix(self, y, x, tp, fp, tn, fn):
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        self.axes[y, x].imshow(confusion_matrix, cmap="Blues")
        self.axes[y, x].set_title("Confusion Matrix")

        labels = ["0", "1"]
        num_classes = len(labels)
        tick_marks = np.arange(num_classes)

        self.axes[y, x].set_xticks(tick_marks, labels)
        self.axes[y, x].set_yticks(tick_marks, labels)
        self.axes[y, x].set_xlabel("Predicted")
        self.axes[y, x].set_ylabel("Actual")

    def plot_f1_epoch_score_curve(self, y, x, epochs, f1_scores):
        self.axes[y, x].clear()
        self.axes[y, x].plot(epochs, f1_scores)
        self.axes[y, x].set_title("F1 Score/Epochs Curve")
        self.axes[y, x].set_xlabel("Epochs")
        self.axes[y, x].set_ylabel("F1 Score")
        self.axes[y, x].set_ylim([0, 1])

    def plot_f1_threshold_score_curve(self, y, x, threshold, f1_scores):
        self.axes[y, x].clear()
        self.axes[y, x].plot(threshold, f1_scores)
        self.axes[y, x].set_title("F1 Score/Threshold Curve")
        self.axes[y, x].set_xlabel("Thresholds")
        self.axes[y, x].set_ylabel("F1 Score")
        self.axes[y, x].set_ylim([0, 1])
        self.axes[y, x].set_xlim([0, 1])

    def plot_accuracy_threshold_curve(self, y, x, threshold, acc):
        self.axes[y, x].clear()
        self.axes[y, x].plot(threshold, acc)
        self.axes[y, x].set_title("Learning Curve")
        self.axes[y, x].set_xlabel("Thresholds")
        self.axes[y, x].set_ylabel("Accuracy")
        self.axes[y, x].set_ylim([0, 1])
        self.axes[y, x].set_xlim([0, 1])


if __name__ == "__main__":
    import time

    p = Myplot()

    while True:
        epochs = [1, 2, 3, 4, 5, 6]
        train_accuracy = p.generate_random_list(0, 1)
        val_accuracy = p.generate_random_list(0, 1)
        p.plot_accuracy_curve(epochs, train_accuracy, val_accuracy)
        p.plot_confusion_matrix(0.8, 0.2, 0.6, 0.1, 0.2, 0.3)
        p.plot_learning_curve(epochs, train_accuracy, val_accuracy)
        p.plot_roc_curve(train_accuracy, val_accuracy)
        p.plot_f1_epoch_score_curve(epochs, train_accuracy)
        p.fresh()

        time.sleep(1)
