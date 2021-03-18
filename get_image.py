import cv2
import time
import sys
import os


def main():
    # ディレクトリの作成
    output_dir = "image_outputs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 動画を読み込む
    video = cv2.VideoCapture("1397552685.mp4")
    # 識別器を読み込む
    cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")

    # 読み込めなかった場合、強制終了する
    if not video.isOpened():
        print("failed")
        sys.exit()

    framecount = -1
    imagecount = 0
    while True:
        framecount += 1
        ret, frame = video.read()
        if framecount % 10 != 0:
            continue
        # 処理を高速化するためにグレースケールにする
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
        # 適用的ヒストグラム平坦化を行う
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(gray_frame)
        """
        # 顔の検出　返される形は[[x,y,w,h],[x,y,w,h]..]
        face = cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        # 顔が存在した場合に保存を行う
        if len(face) != 0:
            for i, (x, y, w, h) in enumerate(face):
                face_image = frame[y:y + h, x:x + w]
                imagename = "image" + str(imagecount) + ".png"
                imagepath = os.path.join(output_dir, imagename)
                cv2.imwrite(imagepath, face_image)
                imagecount += 1
        if imagecount > 200:
            break

    video.release()


if __name__ == "__main__":
    main()
