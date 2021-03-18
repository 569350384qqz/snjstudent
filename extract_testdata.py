from call_me_keras import Neural_NetWork
from gotiusa_keras import Processing_Image
import os
import cv2
import sys


def make_direktory():
    for i in range(6):
        name_dir = "images_test" + str(i)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)


def main():
    make_direktory()
    network = Neural_NetWork()
    process = Processing_Image()
    model = network.constract_network()
    model.load_weights("bigdata_gotiusa_256_e20.h5")
    # 動画を読み込む
    video = cv2.VideoCapture("sm35410243.mp4")
    # 識別器を読み込む
    cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    # 読み込めなかった場合、強制終了する
    if not video.isOpened():
        print("failed")
        sys.exit()
    framecount = -1
    imagecount = 0
    list_numofimage = [0 for i in range(6)]

    while True:
        framecount += 1
        ret, frame = video.read()
        if framecount % 7 != 0:
            continue
        # 処理を高速化するためにグレースケールにする
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 顔の検出　返される形は[[x,y,w,h],[x,y,w,h]..]
        face = cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        # 顔が存在した場合に保存を行う
        if len(face) != 0:
            for i, (x, y, w, h) in enumerate(face):
                face_image = frame[y:y + h, x:x + w]
                image = face_image
                height, width, ch = image.shape
                # 画像の大きさが一律でないため、大きさを揃える
                if height >= 100 or width >= 100:
                    image = cv2.resize(image, dsize=(100, 100),
                                       interpolation=cv2.INTER_AREA)
                else:
                    image = cv2.resize(image, dsize=(100, 100),
                                       interpolation=cv2.INTER_LINEAR)
                height, width, ch = image.shape
                import numpy as np
                norm = np.zeros((height, width))
                # 画像の正規化
                images = cv2.normalize(
                    image, norm, 0, 1, norm_type=cv2.NORM_MINMAX)
                prediction = model.predict(np.array([images]))
                prediction = prediction.tolist()[0]
                max_index = prediction.index(max(prediction))
                imagename = "image" + str(list_numofimage[max_index]) + ".png"
                output_dir = "images_test" + str(max_index)
                imagepath = os.path.join(output_dir, imagename)
                #cv2.imwrite(imagepath, face_image)
                list_numofimage[max_index] += 1
                imagename = "image" + str(imagecount) + ".png"
                #imagepath = os.path.join(output_dir, imagename)
                cv2.imwrite(imagepath, face_image)
                imagecount += 1
        if imagecount > 20000:
            print("sucess")
            break


if __name__ == "__main__":
    main()
