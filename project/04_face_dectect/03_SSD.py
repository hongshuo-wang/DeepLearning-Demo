import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./face_image/face1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用下面这个链接下载对应的预训练权重
# https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detector/deploy.prototxt
# https://github.com/Shiva486/facial_recognition/blob/master/res10_300x300_ssd_iter_140000.caffemodel
# 加载模型
face_detector = cv2.dnn.readNetFromCaffe('./weight/deploy.prototxt',
                                             './weight/res10_300x300_ssd_iter_140000.caffemodel')

# 检测人脸
# 原图尺寸
img_height = img.shape[0]
img_width = img.shape[1]

img_resize = cv2.resize(img, (500, 300))

# 图像转为二进制
img_blob = cv2.dnn.blobFromImage(img_resize, 1.0, (500, 300), (104.0, 177.0, 123.0))

# 输入
face_detector.setInput(img_blob)
# 推理，返回格式【1， 1， face_num, location】
detections = face_detector.forward()

# 查看人脸数量
num_of_faces = detections.shape[2]

# 原图复制，一会儿绘制用
img_copy = img.copy()
for index in range(num_of_faces):
    # 置信度
    detection_confidence = detections[0, 0, index, 2]
    # 挑选置信度
    if detection_confidence > 0.15:
        # 位置(由于缩放过，因此要恢复到原图的尺寸人脸对应的位置)
        locations = detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
        lx, ly, rx, ry = locations.astype('int')
        # 绘制
        cv2.rectangle(img_copy, (lx, ly), (rx, ry), (0, 255, 0), 5)


plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.show()
