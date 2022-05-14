import cv2
import matplotlib.pyplot as plt
# 安装dlib包
# conda install -c conda-forge dlib
import dlib

img = cv2.imread('./face_image/face1.png')

# 构造HOG人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()

# 检测人脸
# upsample=1, 上采样1次，图像较大时，多上采样可以检测到更小的人脸
detections = hog_face_detector(img, 1)

# 解析人脸
for face in detections:
    x = face.left()
    y = face.top()
    r = face.right()
    b = face.bottom()
    cv2.rectangle(img, (x, y), (r, b), (0, 255, 255), 5)
plt.imshow(img)
plt.show()
