import matplotlib.pyplot as plt

def read_log_file(log_file_path):
    epochs = []
    losses = []
    accuracies = []
    with open(log_file_path, 'r') as f:
        epoch = 0
        loss = 0
        accuracy = 0
        for line in f:
            if line.startswith("Epoch"):
                epoch = int((line.split(" [")[1]).split("/")[0])
                if line.split(" ")[2] == "Loss:":
                    loss = float(line.split(" ")[3])
            elif line.startswith("Accuracy"):
                accuracy = float(line.split(" ")[4])

            if accuracy != 0:
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(accuracy)
                accuracy = 0

        return epochs, losses, accuracies


def plot_loss_and_accuracy(epochs, losses, accuracies):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, losses, marker='o')
    plt.ylabel('Loss')
    plt.title('Loss---Epoch')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracies, marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy---Epoch')

    plt.tight_layout()
    plt.show()

log_file_path = './log.txt'
epochs, losses, accuracies = read_log_file(log_file_path)
plot_loss_and_accuracy(epochs, losses, accuracies)
