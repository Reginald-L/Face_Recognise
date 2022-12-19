import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils import get_all_embeddings, cal_euDistance
from facenet_pytorch import MTCNN, InceptionResnetV1


# 1. 构造MTCNN和InceptionResnetV1对象
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

base_url = os.getcwd()

# 2. 读取图像
img_path = os.path.join(base_url, 'data', 'linghua.jpg')
img = Image.open(img_path)
plt.imshow(img)
plt.show()

# 3. 获取人脸
face_cropped, probs, boxes = mtcnn(img, save_path=None, return_prob=True)
print(f'get face with probability {probs}')

print(f'boxes = {boxes}')

# 4. 构造人脸特征
face_embeddings = []
for item in face_cropped:
    face_embedding = resnet(item.unsqueeze(0))
    face_embeddings.append(face_embedding)


# 5. 从数据库中加载存在的人脸特征向量
embedding_path = os.path.join(base_url, 'embeddings')
embeddings = get_all_embeddings(embedding_path) # list [{name, embedding}]

frame_draw = img.copy()
draw = ImageDraw.Draw(frame_draw)

# 6. 计算欧式距离
threshold = 1.0
for i, face_embedding in enumerate(face_embeddings):
    print(i)
    result = ''
    for embedding in embeddings:
        emb_1 = embedding['embedding']
        name = embedding['name']
        dis = cal_euDistance(emb_1, face_embedding)
        print(f'距离 = {dis}')
        if dis < threshold:
            result = name
    print(f'name = {result}')
    # 7. 绘制人脸边框和标签
    box = boxes[i]
    print(f'box = {box}')
    draw.rectangle(box, outline=(0, 255, 0))
    result = result if result != '' else 'Other'
    draw.text((box[0], box[1] - 10), result)

plt.imshow(frame_draw)
plt.show()