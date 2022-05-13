import cv2
from cv2 import circle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time

def showImage(img=[], num=[111]):
    for i in range(len(num)):
        plt.subplot(num[i])
        plt.imshow(img[i])
    plt.show()

def showImageWithCv2():
    # 读取照片
    img = cv2.imread("test.png")

    while True:
        # 显示图片
        cv2.imshow("Demo", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()


def drawBasic():
    black_img = np.zeros((800, 800, 3), dtype=np.int16)
    rectangle_image = black_img.copy()
    cv2.rectangle(rectangle_image, (234, 345), (700, 700), (0, 218, 10), 10)
    circle_image = black_img.copy()
    cv2.circle(circle_image, (122, 333), 100, (0, 0, 233), 10)
    line_image = black_img.copy()
    cv2.line(line_image, (0, 11), (800, 800), (200, 0, 0), 10)
    text_image = black_img.copy()
    cv2.putText(text_image, "python", (500, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 5, cv2.LINE_AA)
    showImage([rectangle_image, circle_image, line_image, text_image], [221, 222, 223, 224])


def showVideo():
    cap = cv2.VideoCapture(0)
    while True:
        rec, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Demo", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

def writeVideo():
    cap = cv2.VideoCapture(0)

    # DIVX, X264, mp4v
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    # 要先创建好文件夹，否则不成功
    writer = cv2.VideoWriter('./video/test.mp4', fourcc, fps, (width, height))

    while True:
        rec, frame = cap.read()
        frame = cv2.flip(frame, 1)
        grayt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Demo", frame)
        # 写入视频
        writer.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    writer.release()
    cap.release()

def readvideo():
    cap = cv2.VideoCapture("./video/test.mp4")
    if not cap.isOpened():
        print('文件不存在或者编码错误')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

def showFps():
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        now = time.time()
        fps_time = int(1/(now-start_time))
        start_time = now
        cv2.putText(frame, str(fps_time), (120, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 100), 10)
        cv2.imshow("Demo", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # showImageWithCv2()
    # drawBasic()
    # showVideo()
    # writeVideo()
    readvideo()
    # showFps()
