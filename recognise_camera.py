import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import get_all_embeddings, cal_euDistance
from facenet_pytorch import MTCNN, InceptionResnetV1


# 1. 构造MTCNN和InceptionResnetV1对象
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

base_url = os.getcwd()

# 2. 从数据库中加载存在的人脸特征向量
embedding_path = os.path.join(base_url, 'embeddings')
embeddings = get_all_embeddings(embedding_path) # list [{name, embedding}]

# 3. 打开摄像头读取图像
camera = cv.VideoCapture(0)
camera.set(3, 1920) # 设置窗口宽
camera.set(4, 1080) # 设置窗口高

# 4. 获取人脸
def step4(img):
    face_cropped, probs, boxes = mtcnn(img, save_path=None, return_prob=True)
    print(f'get face with probability {probs}')
    print(f'boxes = {boxes}')
    return face_cropped, boxes

# 5. 构造人脸特征
def step5(face_cropped):
    print(f'开始构造人脸特征')
    face_embeddings = []
    for item in face_cropped:
        face_embedding = resnet(item.unsqueeze(0))
        face_embeddings.append(face_embedding)
    return face_embeddings

# 6.计算欧式距离
def step6(face_embeddings, embeddings, boxes):
    print(f'开始计算欧式距离, 并进行绘制人脸框和标签')
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
        cv.rectangle(cam, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=3)
        result = result if result != '' else 'Other'
        cv.putText(cam, result, (box[0], box[1] - 10), fontFace=cv.FONT_ITALIC, fontScale=1, color=(0, 0, 255), thickness=2)
        print(f'绘制结束')
    return cam

while True:
    try:
        success, cam = camera.read()
        
        cam = cv.cvtColor(cam, cv.COLOR_BGR2RGB)

        face_cropped, boxes = step4(cam)

        # 5. 构造人脸特征
        if len(face_cropped) != 0:
            face_embeddings = step5(face_cropped)

            # 6. 计算欧式距离
            cam = step6(face_embeddings, embeddings, boxes)

        cam = cv.cvtColor(cam, cv.COLOR_RGB2BGR)
        cv.imshow('MyCamera', cam)
        
    except IndexError:
        print(f'等待人脸进入检测区域')
        success, cam = camera.read()
        cv.imshow('MyCamera', cam)
    finally:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# 释放摄像头
camera.release()

