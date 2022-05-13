import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# PIL读取图片
img = Image.open("test.png")
# PIL图片转换为numpy数组
img_arr = np.array(img)
# 提取B通道
channel_B = img_arr[:,:,2]
# 从numpy数组变为PIL图片
pil_image = Image.fromarray(img_arr)
# 显示图像
plt.imshow(img_arr[:, :, 2]) 
plt.show()
# 或者
pil_image.show("test")
