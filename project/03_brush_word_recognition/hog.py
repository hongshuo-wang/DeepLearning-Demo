import cv2
import matplotlib.pyplot as plt


# 安装skimage
# conda install scikit-image
from skimage.feature import hog
from skimage import exposure

img = cv2.imread('./test_imgs/cat.png')
img = cv2.resize(img, (200, 180))

img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
image: 输入图像
orientation: 把180度分成几份
poxels_per_cell: 元组形式，一个cell内的像素的大小
cells_per_block: 元组形式，一个block内的cell大小
cisualize: 是否需要可视化，如果True，hog会返回numpy图像
'''
fd, hog_image = hog(image=img_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_fixed)
ax1.set_title('Input image')

# 增强显示效果
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax1.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()
