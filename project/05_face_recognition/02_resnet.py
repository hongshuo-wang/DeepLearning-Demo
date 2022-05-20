import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import glob
from sklearn.metrics import accuracy_score


def test01():
    # 人脸检测
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测模型
    # http://dlib.net/files/ 在这里下载模型
    shape_detector = dlib.shape_predictor('./weight/shape_predictor_68_face_landmarks.dat')

    # 读取一张测试图片
    img = cv2.imread('./face_image/face1.png')
    # 检测人脸
    detections = hog_face_detector(img, 2)
    # 对每个人脸进行关键点提取
    for face in detections:
        # 人脸框坐标
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        # 获取68个关键点
        points = shape_detector(img, face)
        # 绘制关键点(用内置方法parts()才可以迭代)
        for point in points.parts():
            cv2.circle(img, (point.x, point.y), 2, (0, 255, 255), -1)

        # 绘制人脸框
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def getFaceFeatLabel(fileName):
    # 获取人脸Label ID（从文件名下手,拿到数字:要6不要06，故取整）
    # subject01.normal.gif
    face_id = int(fileName.split('/')[-1].split('.')[0].split('subject')[-1])

    # 读取图片
    cap = cv2.VideoCapture(fileName)
    ret, img = cap.read()
    # 转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 人脸检测
    detections = hog_face_detector(img, 2)
    face_descriptor = None   # 默认描述符是None，表示没有识别到人脸
    for face in detections:
        # 获取关键点
        points = shape_detector(img, face)
        # 获取特征描述符
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(img, points)
        # resnet只接收np类型的数据，此时的face_descriptor是dlib内置的类型，要转换后再使用
        face_descriptor = [i for i in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = np.reshape(face_descriptor, (1, -1))  # 变成1行128列
    return face_id, face_descriptor


if __name__ == '__main__':
    # test01()

    # 人脸检测
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测模型
    # http://dlib.net/files/ 在这里下载模型
    shape_detector = dlib.shape_predictor('./weight/shape_predictor_68_face_landmarks.dat')
    # resnet模型
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weight/dlib_face_recognition_resnet_model_v1.dat')
    # 训练模型
    file_list = glob.glob('./yalefaces/train/*')
    # 标签列表
    label_list = []
    # 特征描述符列表
    feature_list = None
    # 文件列表
    name_list = {}
    index = 0
    for train_file_path in file_list:
        # 获取每一张图片的对应信息
        label, feature = getFaceFeatLabel(train_file_path)
        # 过滤数据
        if feature is not None:
            # 检测到了人脸
            name_list[index] = train_file_path
            label_list.append(label)
            if feature_list is None:
                feature_list = feature
            else:
                feature_list = np.concatenate((feature_list, feature), axis=0)
            index += 1
    # 评估测试集
    test_list = glob.glob('./yalefaces/test/*')
    # 预测结果
    predict_list = []
    label_list = []
    # 距离阈值
    threshold = 0.5
    for test_file in test_list:
        # 获取每一张图片的对应信息
        label, feature = getFaceFeatLabel(test_file)
        # 过滤数据
        if feature_list is None:
            # 计算所有距离(取list中其中一个值和整个list中每一个元素进行计算)
            distances = np.linalg.norm((feature-feature_list), axis=1)
            # 最短距离的索引
            min_index = np.argmin(distances)
            # 拿到最短距离
            min_distance = distances[min_index]

            if min_distance < threshold:
                # 同一个人
                predict_id = int(name_list[min_distance].split('/')[-1].split('.')[0].split('subject')[-1])
            else:
                predict_id = -1
            predict_list.append(predict_id)
            label_list.append(label)
    print(accuracy_score(label_list, predict_list))
