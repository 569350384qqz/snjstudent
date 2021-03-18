from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


class Processing_Image:
    # 画像の読み込みを行い、リストに保存する
    def loadimage(self):
        image_list = []
        for i in range(202):
            count = 0
            image = cv2.imread(
                "image_changed_remaster/image" + str(i) + ".png")
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
            image = cv2.normalize(image, norm, 0, 1, norm_type=cv2.NORM_MINMAX)
            image_list.append(np.array(image))
        return image_list

    # ラベルデータ（csv）から正解ラベルを示す行列に変換
    def load_trainlabel(self, data):
        list_trainlabel = []
        for i in data:
            label_base = [0 for i in range(6)]
            label_base[int(i)] = 1
            list_trainlabel.append(label_base)
        list_trainlabel = np.array(list_trainlabel)
        return list_trainlabel


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
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(6, activation="sigmoid"))
        # Adadeltaいいっぽい
        model.compile(optimizer="Adadelta",
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def calculate_network(self, model, x_train, y_train, x_test, y_test):
        history = model.fit(x_train, y_train, epochs=10,
                            batch_size=1, validation_data=(x_test, y_test))
        model.save("mindata_gotiusa.h5")
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


def main():
    Process = Processing_Image()
    image_list = Process.loadimage()
    data = np.loadtxt("train.csv", delimiter=",")
    trainlabel = Process.load_trainlabel(data)
    x_train, x_test, y_train, y_test = image_list[:
                                                  180], image_list[180:], trainlabel[:180], trainlabel[180:]
    Neural_NetWorks = Neural_NetWork()
    model = Neural_NetWorks.constract_network()
    history = Neural_NetWorks.calculate_network(
        model, np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
    Neural_NetWorks.plot_histrory(history)


if __name__ == "__main__":
    main()
