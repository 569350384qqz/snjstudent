from gotiusa_keras import Processing_Image
from call_me_keras import Neural_NetWork
import os
import cv2
import sys


def main():
    network = Neural_NetWork()
    process = Processing_Image()
    model = network.constract_network()
    model.load_weights("bigdata_gotiusa_256_e20.h5")
    # 動画を読み込む
    video = cv2.VideoCapture("no-poi_op.mp4")
    # 識別器を読み込む
    cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, video.get(
        cv2.CAP_PROP_FPS), (1280, 720))
    # 読み込めなかった場合、強制終了する
    if not video.isOpened():
        print("failed")
        sys.exit()
    while (video.isOpened()):
        ret, frame = video.read()

        try:
            # 処理を高速化するためにグレースケールにする
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
        #height_f, width_f, ch_f = frame.shape
        #print(height_f, width_f)
        # sys.exit()
        # 顔の検出　返される形は[[x,y,w,h],[x,y,w,h]..]
        face = cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        # 顔が存在した場合に保存を行う
        if len(face) != 0:
            for i, (x, y, w, h) in enumerate(face):
                face_image = frame[y:y + h, x:x + w]
                image = face_image
                height, width, ch = image.shape
                import numpy as np
                norm = np.zeros((height, width))
                # 画像の大きさが一律でないため、大きさを揃える
                if height >= 100 or width >= 100:
                    images = cv2.resize(image, dsize=(100, 100),
                                        interpolation=cv2.INTER_AREA)
                else:
                    images = cv2.resize(image, dsize=(100, 100),
                                        interpolation=cv2.INTER_LINEAR)
                # 画像の正規化
                images = cv2.normalize(
                    images, norm, 0, 1, norm_type=cv2.NORM_MINMAX)
                prediction = model.predict(np.array([images]))
                prediction = prediction.tolist()[0]
                max_index = prediction.index(max(prediction))
                if max_index == 0:
                    color = (147, 88, 120)
                elif max_index == 1:
                    color = (190, 165, 245)
                elif max_index == 2:
                    color = (255, 150, 79)
                elif max_index == 3:
                    color = (111, 180, 141)
                elif max_index == 4:
                    color = (161, 215, 244)
                elif max_index == 5:
                    color = (0, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
        out.write(frame)
    video.release()
    out.release()


if __name__ == "__main__":
    main()
