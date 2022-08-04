# -*- coding:utf-8 -*-
# Author: yuyangmu
# Description: 通过 faiss 进行余弦相似度检索
# Date: 2022-07-30


import time
import faiss
from faiss import normalize_L2
import pandas as pd
import numpy as np
import pickle as pk
from typing import List, Tuple, Union

from sentence_transformers import SentenceTransformer
from torchsummary import summary


def read_to_pick(xlsx_file: str, pick_file: str) -> None:
    df = pd.read_excel(xlsx_file)
    data1 = pd.DataFrame(df, columns=["主问法"]).dropna(axis=0)
    data2 = pd.DataFrame(df, columns=["次问法"]).dropna(axis=0)
    data = np.array(data1).tolist() + np.array(data2).tolist()
    set_data = set()
    for item in data:
        for text in item:
            set_data.add(text)
    print(len(set_data), list(set_data))
    # 把标准问的list写入到二进制文件中
    with open(pick_file, "wb") as write_file:
        pk.dump(list(set_data), write_file)


def get_faiss_index(qt2vec_file: str, type: str = "L2"):
    with open(qt2vec_file, "rb") as file:
        qt2vec = pk.load(file)

    k = 0
    id2qt = dict()
    vectors = list()
    for question, vec in qt2vec.items():
        id2qt[k] = question
        vectors.append(vec)
        k += 1
    vectors = np.array(vectors).astype('float32')

    # 构建索引
    index = faiss.IndexIDMap(faiss.IndexFlatL2(len(vectors[0])))
    index.add_with_ids(vectors, np.arange(0, len(vectors)))

    if type == "COS_SIM":
        normalize_L2(vectors)
        # 构建索引
        index = faiss.IndexIDMap(faiss.IndexFlatIP(len(vectors[0])))
        index.add_with_ids(vectors, np.arange(0, len(vectors)))

    return id2qt, index


class IndexOfL2(object):
    def __init__(self, id2qt, index):
        self.id2qt = id2qt
        self.index = index

    def search(self, query_vector: Union[List[float], List[List[float]]], k: int = 1,
               type: str = "L2") -> List[Tuple]:
        """
        :param query_vector: Union[List[float], List[List[float]]]
        :param k: int
        :param type: str
        :return: List[Tuple[Any, Any]]
        """
        if not isinstance(query_vector[0], list):
            query_vector = [query_vector]

        query_vector = np.array(query_vector).astype('float32')
        if type == "COS_SIM":
            normalize_L2(query_vector)

        t = time.time()
        top_k_scores, top_k_ids = self.index.search(query_vector, k)
        print("total time of search top k:", (time.time() - t) / 1000)

        results = []
        for top_k_score, top_k_id in zip(top_k_scores[0], top_k_ids[0]):
            results.append((top_k_score, self.id2qt[top_k_id]))

        return results


if __name__ == "__main__":
    k = 2
    is_type = "COS_SIM"
    query = "会员如何自动续费？"

    model = SentenceTransformer('./model/training_similarity_model_2021-12-24_15-36-08')
    summary(model)

    qt2Vec_file = "./model/qt2vec.txt"
    id2Qt, faiss_index = get_faiss_index(qt2Vec_file, is_type)

    query_vec = model.encode(query)
    indexL2 = IndexOfL2(id2Qt, faiss_index)
    results = indexL2.search(query_vec.tolist(), k, is_type)
    print(results)

