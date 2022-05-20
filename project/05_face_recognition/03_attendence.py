import cv2
import dlib


def faceRegister(label_id=1, name="harrison", count=3, interval=1):
    """
    人脸注册方法
    :param label_id: 序号
    :param name: 人脸名称
    :param count: 采集数量
    :param interval: 采集间隔时间
    :return:
    """
    cap = cv2.VideoCapture(0)

    # 获取长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()
    # 构造人脸68个关键点检测器
    shape_detector = dlib.shape_predictor('./weight/shape_predictor_68_face_landmarks.dat')

    while cap.isOpened():
        ret, frame = cap.read()
        # 缩放
        # frame = cv2.resize(frame, (width//2, height//2))
        # 镜像
        frame = cv2.flip(frame, 1)
        # 转为灰度图
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        detections = hog_face_detector(frame, 1)  # 解析人脸
        for face in detections:
            # 获取人脸坐标
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            # 获取人脸关键点
            points = shape_detector(frame, face)
            # 绘制关键点
            for point in points.parts():
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 65), -1)
            # 绘制人脸框
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 67), 2)

        # 显示画面
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def faceRecognizer():
    pass


if __name__ == "__main__":
    faceRegister()
