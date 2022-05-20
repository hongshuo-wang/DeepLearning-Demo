import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import glob

# 用人脸检测器将人脸提取出来
hog_face_detector = dlib.get_frontal_face_detector()


# 图片预处理
def getFaceImgLabel(fileName):
    cap = cv2.VideoCapture(fileName)
    ret, img = cap.read()
    # 转为灰度图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    detections = hog_face_detector(img, 1)
    # 判断是否有人脸
    if len(detections) > 0:
        # 一张图只有一个人脸，所以不用for循环，只拿第一个人脸即可
        face_location = detections[0]
        # 获取人脸坐标
        x = face_location.left()
        y = face_location.top()
        r = face_location.right()
        b = face_location.bottom()
        # 截取人脸
        face_crop = img[y:b, x:r]
        # 获取人脸Label ID（从文件名下手,拿到数字:要6不要06，顾取整）subject01.normal.gif
        face_id = int(fileName.split('/')[-1].split('.')[0].split('subject')[-1])
        return face_crop, face_id
    else:
        return None, -1


# 对多个数据进行预处理
def preprocessDataset(datasetPath):
    face_list = []
    id_list = []
    file_list = glob.glob('./yalefaces/train/*')
    for train_file in file_list:
        # 获取每一张图片的人脸和ID
        face_img, face_id = getFaceImgLabel(train_file)
        # 过滤数据
        if face_id != -1:
            face_list.append(face_img)
            id_list.append(face_id)
    return face_list, id_list

if __name__ == "__main__":
    # 图片与处理，获取人脸图片和id
    face_list, id_list = preprocessDataset('./yalefaces/train/*')

    # 构造分类器
    face_cls = cv2.face.LBPHFaceRecognizer_create()
    # face_cls = cv2.face.EigenFaceRecognizer_create()
    # face_cls = cv2.face.FisherFaceRecognizer_create()

    # 训练(要求图片和label都是np.array格式的数据)
    face_cls.train(face_list, np.array(id_list))
    # 预测
    result = face_cls.

