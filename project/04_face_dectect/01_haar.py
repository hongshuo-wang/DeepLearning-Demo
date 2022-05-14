import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./face_image/face1.png")
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 构造haar检测器
# 权重下载路径：anaconda3\envs\DL\Lib\site-packages\cv2\data
face_detector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')
# 转为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(img_gray)
print(detections)
# 解析检测结果
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 23), 5)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 调节一些参数使效果更好
# scaleFactor: 必须大于1，默认1.1，越大检测到的人脸越大（小的检测不到）
# minNeighbors: 候选人脸数量，越多代表当前图像要想被识别为人脸，需要有很多候选框都在附近
# minSize: 最小人脸尺寸(w, h)
detections = face_detector.detectMultiScale(img_gray,
                                            scaleFactor=1.3, minNeighbors=7, minSize=(5, 5), maxSize=(10, 10))
# 解析检测结果
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 23), 5)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
