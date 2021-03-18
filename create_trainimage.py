import glob
import cv2
import os

for i in range(6):
    file_name = "images_" + str(i)
    file_name_missed = "images_" + str(i) + "_missed"
    file_name_gen = "images_generated_" + str(i)
    train_file_name = "images_train_gen_" + str(i)
    accurate_file = glob.glob(file_name + "/image*")
    missed_file = glob.glob(file_name_missed + "/image*")
    generated_file = glob.glob(file_name_gen + "/image*")
    count = 0
    if not os.path.exists(train_file_name):
        os.mkdir(train_file_name)
    for u in accurate_file:
        image = cv2.imread(u)
        imagename = "image_" + str(i) + "_" + str(count)+".png"
        outputpath = os.path.join(train_file_name, imagename)
        cv2.imwrite(outputpath, image)
        count += 1

    for u in missed_file:
        image = cv2.imread(u)
        outputpath = os.path.join(
            train_file_name, "image_" + str(i) + "_" + str(count)+".png")
        cv2.imwrite(outputpath, image)
        count += 1
    for u in generated_file:
        image = cv2.imread(u)
        outputpath = os.path.join(
            train_file_name, "image_" + str(i) + "_" + str(count)+".png")
        cv2.imwrite(outputpath, image)
        count += 1
