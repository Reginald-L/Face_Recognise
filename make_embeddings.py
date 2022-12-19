import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle


# 1. 构造MTCNN对象和InceptionResnetV1对象
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 2. 加载人脸数据
base_path = os.getcwd()
basedata_path = os.path.join(base_path, 'database')
imgs = os.listdir(basedata_path)

# 3. 识别人脸
for img_path in imgs:
    abs_img_path = os.path.join(basedata_path, img_path)
    img = Image.open(abs_img_path)
    # 获取截取到的人脸
    img_cropped, probs = mtcnn(img, save_path=None, return_prob=True)
    # 压缩为512维向量
    embedding = resnet(img_cropped.unsqueeze(0))
    # 将json数据写入basedata.json文件中
    file_name = img_path.split('.')[0]
    embedding_path = os.path.join(base_path, 'embeddings', '%s.txt'%(file_name))
    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding, f)

