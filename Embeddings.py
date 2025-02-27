import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# 将ca证书环境变量置为空
os.environ['CURL_CA_BUNDLE'] = ''
from dotenv import load_dotenv, find_dotenv
# 加载.env文件 api_key
_ = load_dotenv(find_dotenv())

class BaseEmbeddings:
    
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
        
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
       
    @classmethod
    def cosine_similarity(cls, a: List[float], b: List[float]) -> float:
        dot_product = np.dot(a, b)
        # 向量长度 magnitude
        magnitude = np.linalg.norm(a) + np.linalg.norm(b)
        # 如果有0向量，默认相似度为0，返回0
        if not magnitude:
            return 0
        return dot_product / magnitude
        
class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, str, is_api)
        # 创建 OpenAI 实例
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
            if self.is_api:
                text = text.replace("\n", " ")
                # 传参要求是列表，所以用 [text]，返回值也是列表
                return self.client.embeddings.create(input=[text], model=model).data[0].embedding
            else:
                raise NotImplementedError
                
                
