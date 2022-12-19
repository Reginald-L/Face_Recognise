import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import numpy as np


# mtcnn = MTCNN(keep_all=True)
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open('data/mxt_2.jpg')
plt.imshow(img)
plt.show()

save_path = os.getcwd()
# save_path = os.path.join(save_path, 'linghua_face.jpg')
img_cropped, probs, boxes = mtcnn(img, save_path=None, return_prob=True) # 直接调用mtcnn中的forward函数, 返回截取到的脸部和概率
print(f'img_cropped.size = {img_cropped.shape}')

def show_img_cropped(img_cropped):
    if img_cropped is not None:
        # img_cropped_pil = torch.tensor(img_cropped[0]).permute(1, 2, 0) # 和MTCNN(keep_all=True)匹配使用
        img_cropped_pil = torch.tensor(img_cropped).permute(1, 2, 0)
        print(f'img_cropped_pil.shape = {img_cropped_pil.shape}')
        plt.imshow(img_cropped_pil)
        plt.show()
        print(f'img_cropped with prob {probs}')

show_img_cropped(img_cropped)

# img_detect 返回的是检测到的所有脸的box(x1, y1, x2, y2)
# probs 返回的是检测到的每个脸部信息的可信度
# points 返回的是每个脸部信息中的5个核心点: 左眼, 右眼, 鼻子, 左嘴角, 右嘴角
img_detect, probs, points = mtcnn.detect(img, landmarks=True) 
print(f'img_detect = {img_detect}')

# 定义一个ImageDraw对象在图像中进行绘制识别到的脸部
frame_draw = img.copy()
draw = ImageDraw.Draw(frame_draw)

print(f'points = {points.shape}')
# 遍历所有的脸部信息对其绘制边框
if points.shape[0] > 1:
    # 检测到多个人脸
    print('检测到多个人脸')
    for i, item in enumerate(zip(img_detect, points.squeeze())):
        box, points = item
        print(f'size = {box[2] - box[0]}')
        # left, top, right, bottom (x1, y1, x2, y2)
        draw.rectangle(box.tolist(), outline=(0, 255, 0))
        # points = points.tolist()
        print(points.shape)
        print(f'points = {points}')
        for point in points:
            draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill=(0, 255, 255))

else:
    # 只检测到一个人脸
    print('只检测到一个人脸')
    box = img_detect[0]
    draw.rectangle(box.tolist(), outline=(0, 255, 0))
    for point in points.squeeze():
        point = point.tolist()
        print(point)
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill=(0, 255, 255))

    
plt.imshow(frame_draw)
plt.show()

# 将脸部信息压缩为一个512维的特征向量
img_embedding = resnet(img_cropped.unsqueeze(0))
# print(f'img_embedding = {img_embedding}')
# print(f'img_embedding.shape = {img_embedding.shape}')

# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
# print(f'img_probs = {img_probs}')
# print(f'img_probs.shape = {img_probs.shape}')

# 第二个人脸图像的读取
img2 = Image.open('data/mxt_3.jpg')
plt.imshow(img2)
plt.show()
img_cropped_2, probs_2, boxes = mtcnn(img2, save_path=None, return_prob=True)
show_img_cropped(img_cropped_2)
img_embedding_2 = resnet(img_cropped_2.unsqueeze(0))
# 计算两个向量的欧式距离
def cal_euDistance(emb_1, emb_2):
    assert emb_1.shape == emb_2.shape, '两个向量的形状不一样'
    dis_sum = 0.0
    for i in range(emb_1.shape[1]):
        x_i = emb_1[0][i]
        y_i = emb_2[0][i]
        z_i = (x_i - y_i)**2
        dis_sum += z_i
    dis = np.sqrt(dis_sum.detach().numpy())
    result = '一个人' if dis < 1 else '不是同一个人'
    return result

result = cal_euDistance(img_embedding, img_embedding_2)
print(result)