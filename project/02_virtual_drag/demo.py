"""
步骤：
1、opencv获取视频流
2、在画面上画一个方块
3、通过mediapipe获取手指关键点坐标
4、判断手指是否在方块上
5、如果在方块上，方块跟着手指移动
"""
import mediapipe as mp
import cv2
import numpy as np
import time
import math
from calculate_distance import calculate_distance_btween_two_fingers as distance

# 视频捕获
cap = cv2.VideoCapture(0)  # 0代表电脑自带的摄像头

# 获取画面的高度和宽度
img_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
img_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 判断手指是否在方块上
on_square = False

# 存储手指相对方块边缘的距离
L1 = 0
L2 = 0

# 设置方块的高宽以及左上角坐标
square_x = 300
square_y = 100
square_width = 100
square_height = 100
square_color = (0, 0, 255)
 
#创建检测手部关键点的方法
mpHands = mp.solutions.hands  #接收方法
hands = mpHands.Hands(static_image_mode=False, #静态追踪，低于0.5置信度会再一次跟踪
                      max_num_hands=1, # 最多有几只手
                      min_detection_confidence=0.5, # 最小检测置信度
                      min_tracking_confidence=0.5)  # 最小跟踪置信度 
 
# 创建检测手部关键点和关键点之间连线的方法
mpDraw = mp.solutions.drawing_utils
# 创建绘图风格
mp_drawing_styles = mp.solutions.drawing_styles
 
# 查看时间
start_time = 0 #处理一张图像前的时间
end_time = 0 #一张图处理完的时间

# 存储食指和中指的坐标
index_finger_x = 0
index_finger_y = 0
middle_finger_x = 0
middle_finger_y = 0
 
#处理视频图像
while cap.isOpened():
    # 记录起始时间
    start_time = time.time()
    
    # 返回是否读取成功和读取的图像
    success, img = cap.read()

    img = cv2.flip(img, 1)

    
    # 在循环中发送rgb图像到hands中，opencv中图像默认是BGR格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 把图像传入检测模型，提取信息
    results = hands.process(imgRGB)
 
    # 检查每帧图像是否有多只手，一一提取它们
    if results.multi_hand_landmarks: #如果没有手就是None
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取每个关键点的索引和坐标
            for index, landmark in enumerate(hand_landmarks.landmark):          
                # 每一个关键点的中心坐标
                center_x , center_y =  int(landmark.x*img_width), int(landmark.y*img_height) #比例坐标x乘以宽度得像素坐标
                # 输出食指8号，中指12号的实际像素点
                if index == 8 or index == 12:
                    # print(f'{index}=>({center_x}, {center_y})')
                    if index == 8:
                        index_finger_x, index_finger_y = center_x, center_y
                    else:
                        middle_finger_x, middle_finger_y = center_x, center_y
            # 判断两个手指是否在方块上
            if (index_finger_x > square_x) and (index_finger_x < square_x + square_width) \
                and (index_finger_y > square_y) and(index_finger_y < square_y + square_height):
                # print('在方块上')
                if distance((index_finger_x, index_finger_y), (middle_finger_x, middle_finger_y)) > 60:
                    # print('不可以移动')
                    on_square = False
                    square_color = (0, 0, 255)
                else:
                    # print('可以移动')
                    square_color = (0, 255, 0)
                    if on_square == False:
                        L1 = abs(index_finger_x - square_x)
                        L2 = abs(index_finger_y - square_y)
                        on_square = True
            else:
                # print('不在方块上')
                on_square = False
                square_color = (0, 0, 255)
            # 如果在方块上，就追踪手指
            if on_square:
                square_x = index_finger_x - L1
                square_y = index_finger_y - L2
                # print(square_x, square_y)

            # 绘制每只手的关键点
            mpDraw.draw_landmarks(img, 
                        hand_landmarks, 
                        mpHands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()) 
                        #传入想要绘图画板img，单只手的信息hand_landmarks
                        # mpHands.HAND_CONNECTIONS绘制手部关键点之间的连线
        
    # 记录执行时间      
    end_time = time.time()      
    # 计算fps
    fps = 1/(end_time-start_time)
    # 重置起始时间
    start_time = end_time    
    overlay = img.copy()
    # 绘制方块
    cv2.rectangle(img, (square_x, square_y), (square_x+square_width, square_y+square_height), square_color, -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    # 把fps显示在窗口上；img画板；取整的fps值；显示位置的坐标；设置字体；字体比例；颜色；厚度
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    # 显示图像
    cv2.imshow('Image', img)  
    if cv2.waitKey(30) & 0xFF==27:  
        break
 
# 释放视频资源
cap.release()
cv2.destroyAllWindows()
