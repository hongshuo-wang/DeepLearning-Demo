import dlib
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./face_image/face1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用下面这个链接下载对应的预训练权重
# https://github.com/davisking/dlib-models
# 构造CNN人脸检测器
cnn_face_detector = dlib.cnn_face_detection_model_v1('./weight/mmod_human_face_detector.dat')

# 检测人脸
# upsample=1, 上采样1次，图像较大时，多上采样可以检测到更小的人脸
detections = cnn_face_detector(img, 1)

# 解析矩形结果
for face in detections:
    x = face.rect.left()
    y = face.rect.right()
    r = face.rect.right()
    b = face.rect.bottom()
    c = face.confidence
    cv2.putText(img, str(c), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 5)
    cv2.rectangle(img, (x, y), (r, b), (0, 234, 43), 5)
    break

plt.imshow(img)
plt.show()
