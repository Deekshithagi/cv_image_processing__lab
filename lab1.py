import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('splitimg.png')
x, y, z = img.shape
x//=2
y//=2

down = img[x:, :]   
up = img[:x, :] 
left = img[:,y:]
right = img[:,:y]

up=cv2.resize(up,(0,0),fx=0.5,fy=.5)
down=cv2.resize(down,(0,0),fx=0.5,fy=.5)
left=cv2.resize(left,(0,0),fx=0.5,fy=.5)
right=cv2.resize(right,(0,0),fx=0.5,fy=.5)

x2, y2, z2 = up.shape
y2//=2

up_right = up[:,y2:]
up_left = up[:,:y2]
down_right = down[:,y2:]
down_left = down[:,:y2]



titles = [
    "up","down","left","right"
]
images = [
    up_left,up_right,down_left,down_right
]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray" if len(images[i].shape) == 2 else None)
    plt.axis("off")

plt.show()

