import os
import pickle
import numpy as np


def get_all_embeddings(embedding_path):
    embeddings = []
    all_embeddings = os.listdir(embedding_path)
    for path in all_embeddings:
        abs_embedding_path = os.path.join(embedding_path, path)
        with open(abs_embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        name = path.split('.')[0]
        embeddings.append({'name': name, 'embedding': embedding})
    return embeddings


def cal_euDistance(emb_1, emb_2):
    assert emb_1.shape == emb_2.shape, '两个向量的形状不一样'
    dis_sum = 0.0
    for i in range(emb_1.shape[1]):
        x_i = emb_1[0][i]
        y_i = emb_2[0][i]
        z_i = (x_i - y_i)**2
        dis_sum += z_i
    dis = np.sqrt(dis_sum.detach().numpy())
    return dis