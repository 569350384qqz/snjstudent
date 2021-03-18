from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import sys
import numpy as np
import glob


def loadimage():
    datagen = ImageDataGenerator(rotation_range=180,
                                 width_shift_range=0.3, height_shift_range=0.3, shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode='nearest')

    for i in range(6):
        trainlabel = [0 for i in range(6)]
        trainlabel[i] = 1
        file_t = glob.glob("images_train_" + str(i) + "/image*")
        save_dir = "images_generated_" + str(i)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for s in range(len(file_t)):
            traingenerator = datagen.flow(
                np.array([cv2.imread(file_t[s])]),
                batch_size=1
            )
            batches = traingenerator
            g_img = batches[0].astype(np.uint8)
            imagename = "images_" + str(i) + "_" + str(s) + ".png"
            output_dir = os.path.join(save_dir, imagename)
            cv2.imwrite(output_dir, g_img[0])


loadimage()
