import cv2

cap = cv2.VideoCapture(0)

# 构造人脸检测器
haar_face_detector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')


while cap.isOpened():
    ret, frame = cap.read()
    # 镜像
    frame = cv2.flip(frame, 1)
    # 转为灰度图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    detections = haar_face_detector.detectMultiScale(frame_gray, minNeighbors=7)
    # 解析人脸
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 23), 5)
    # 显示画面
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

