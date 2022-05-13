import cv2
import numpy as np
import matplotlib.pyplot as plt


# 解决opencv无法读取中文路径的问题
def readImg(filePath):
    raw_data = np.fromfile(filePath, dtype=np.uint8)
    img = cv2.imdecode(raw_data, -1)
    return img

# 图像预处理
def preprocessImg(filePath):
    # 读取
    img = readImg(filePath)
    # 缩放
    img = cv2.resize(img, (100, 100))
    # 灰度
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


img_1 = readImg('./image/楷书/哎/颜真卿_9e2c2e8c9834736ed7cc6e2ac9f9e365de55791f.jpg')
img_1_fixed = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

# plt.imshow(img_1_fixed)
# plt.show()

# 提取Hog特征
from skimage.feature import hog
from skimage import exposure

img_1_resize = cv2.resize(img_1, (200, 200))
img_1_gray = cv2.cvtColor(img_1_resize, cv2.COLOR_BGR2GRAY)
fd, hog_image = hog(image=img_1_gray, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
ax1.imshow(img_1_resize)
ax2.imshow(hog_image)
# plt.show()

# 读取图片(由于数据集样本数量参差不齐，这里统一只取每个字体的1000个随机图片)
import os
import glob
import random

# 将数据集转换后的特征和标签分别存储到两个列表中
feature_list = []
label_list = []

style_list = ['篆书', '草书', '隶书', '行书', '楷书']
for style in style_list:
    print(f'开始处理{style}')
    # 列出该风格下所有图片(返回的是文件名)
    file_path_list = glob.glob('./image/'+ style +'/*/*')
    # 随机打乱文件顺序
    random.shuffle(file_path_list)
    # 只选取其中1000个(文件名)
    selected_file_paths = file_path_list[:500]
    for file_path in selected_file_paths:
        # 拿到并预处理图像
        brush_word_image = preprocessImg(file_path)
        # 对每张图片进行特征提取
        feature = hog(brush_word_image, 4, (10, 10), (4, 4))
        # 获取风格标签
        label = style_list.index(style)
        feature_list.append(feature)
        label_list.append(label)

# 导入SVM模型
from sklearn import svm
from sklearn.model_selection import train_test_split
# 将样本分为训练和测试样本（传入特征和标签,random_state可根据喜好进行设置）
feature_train, feature_test, label_train, label_test = train_test_split(feature_list, label_list, test_size=0.25, random_state=42)

# 训练
cls = svm.SVC()
cls.fit(feature_train, label_train)

# 预测
predicted_labels = cls.predict(feature_test)

# 评估
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(predicted_labels, label_test)
print(accuracy)

# 保存训练好的模型
from joblib import dump, load
if not os.path.exists('models'):
    os.mkdir('models')
dump(cls, './models/brush.pkl')

# 加载模型
new_cls = load('./models/brush.pkl')
new_predicted_labels = new_cls.predict(feature_test)
new_accuracy = accuracy_score(predicted_labels, label_test)
print(new_accuracy)


# 用混淆矩阵看看效果
cm = confusion_matrix(label_test, predicted_labels)
import seaborn as sn
import pandas as pd

df_cm = pd.DataFrame(cm, index=[i for i in ['zhuan', 'cao', 'li', 'xing', 'kai']],
                    columns=[i for i in ['zhuan', 'cao', 'li', 'xing', 'kai']])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Greens', fmt='d')
plt.show()

