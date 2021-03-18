import plaidml.keras
plaidml.keras.install_backend()
from keras import models
from keras import layers
from keras import regularizers
import glob
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np



class Processing_Image:
    # 画像の読み込みを行い、リストに保存する
    def loadimage(self):
        image_list = []
        list_trainlabel = []

        for i in range(6):
            trainlabel = [0 for i in range(6)]
            trainlabel[i] = 1
            file_t = glob.glob("images_train_gen_" + str(i) + "/image*")
            for u in range(len(file_t)):
                image = cv2.imread("images_train_gen_" + str(i) +
                                   "/image_" + str(i) + "_" + str(u) + ".png")

                height, width, ch = image.shape
                # 画像の大きさが一律でないため、大きさを揃える
                if height >= 100 or width >= 100:
                    image = cv2.resize(image, dsize=(100, 100),
                                       interpolation=cv2.INTER_AREA)
                else:
                    image = cv2.resize(image, dsize=(100, 100),
                                       interpolation=cv2.INTER_LINEAR)
                height, width, ch = image.shape
                norm = np.zeros((height, width))
                # 画像の正規化
                image = cv2.normalize(
                    image, norm, 0, 1, norm_type=cv2.NORM_MINMAX)
                image_list.append(np.array(image))
                list_trainlabel.append(trainlabel)
        return image_list, np.array(list_trainlabel)


class Neural_NetWork:
    def constract_network(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu",
                                input_shape=(100, 100, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(
            512, kernel_regularizer=regularizers.l2(0.001), activation="relu"))
        model.add(layers.Dense(6, activation="sigmoid"))
        # Adadeltaいいっぽい
        model.compile(optimizer="Adadelta",
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def calculate_network(self, model, x_train, y_train, x_test, y_test):
        history = model.fit(x_train, y_train, epochs=20,
                            batch_size=256, validation_data=(x_test, y_test))
        model.save("bigdata_gotiusa_256_e20.h5")
        return history

    def plot_histrory(self, history):
        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        epochs = range(1, len(acc)+1)
        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and Validation acciracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def plot_loss(self, history):
        loss = history.history['loss']
        val_loss = history.history['loss']

        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def main():

    Process = Processing_Image()
    image_list, trainlabel = Process.loadimage()
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(min(len(image_list), len(trainlabel))):
        if i % 50 == 0:
            x_test.append(image_list[i])
            y_test.append(trainlabel[i])
        else:
            x_train.append(image_list[i])
            y_train.append(trainlabel[i])

    """
    x_train, x_test, y_train, y_test = image_list[:
                                                  int(len(image_list)*0.9)], image_list[int(len(image_list)*0.9):], trainlabel[:int(len(trainlabel)*0.9)], trainlabel[int(len(trainlabel)*0.9):]
    """
    Neural_NetWorks = Neural_NetWork()
    model = Neural_NetWorks.constract_network()
    history = Neural_NetWorks.calculate_network(
        model, np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
    Neural_NetWorks.plot_loss(history)
    Neural_NetWorks.plot_histrory(history)


if __name__ == "__main__":
    main()
