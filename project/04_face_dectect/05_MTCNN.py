import cv2
import numpy as np
import matplotlib.pyplot as plt
# 安装mtcnn
# conda install -c conda-forge tensorflow
# conda install -c conda-forge mtcnn
from mtcnn.mtcnn import MTCNN

# 读取图片
img = cv2.imread('./face_image/face1.png')
# MTCNN需要RGB通道图片
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载模型
face_detector = MTCNN()

# 检测人脸
detections = face_detector.detect_faces(img_cvt)
for face in detections:
    (x, y, w, h) = face['box']
    cv2.rectangle(img_cvt, (x, y), (x+w, y+h), (0, 255, 0), 5)
plt.imshow(img_cvt)
plt.show()

