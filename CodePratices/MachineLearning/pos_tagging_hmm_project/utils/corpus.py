import numpy as np
from tqdm import tqdm

class CorpusHelper:
    """
    用于读取并枚举语料库的辅助类
    """

    def __init__(self, path, sent_end_token="."):
        """
        param: path, 语料库路径, 类别string
        param: sent_end_token, 句末标点, 类别string
        """
        self.path = path
        self.token2id = {}  # id从0开始
        self.id2token = {}
        self.tag2id = {}  # id从0开始
        self.id2tag = {}
        self.sent_end_token = sent_end_token
        self.prepare_dict()

    def read_lines(self):
        """
        读取数据

        return: token和词性, 类别tuple(类别，词性)
        """
        with open(self.path, "r") as f:
            for line in tqdm(f):
                token, pos_tag = line.strip().split("/")
                yield token, pos_tag

    def read_lines2id(self):
        """
        读取数据，并将token和tag转化为id
        """
        for token, pos_tag in self.read_lines():
            yield self.token2id[token], self.tag2id[pos_tag]

    def is_end_tokenid(self, token_id):
        """
        判断是否句末标点id

        param: token_id 待验证tokenid，类别int
        return: 是否为句末tokenid, 类别bool
        """
        return token_id == self.token2id[self.sent_end_token]

    def id_to_tags(self, ids):
        """
        将id序列转化为词性标注

        param: ids, 待转化词性id，类别list[int]
        return: 词性标注序列, 类别list[string]
        """
        return [self.id2tag[id] for id in ids]

    def id_to_tokens(self, ids):
        """
        将id序列转化为token序列

        param: ids, 待转化id，类别list[int]
        return: token序列, 类别list[string]
        """
        return [self.id2token[id] for id in ids]

    def _update_dict(self, symbol2id, id2symbol, symbol):
        """
        给定新项，更新词典:

        param: symbol2id, 符号id映射词典, 类型dict
        param: id2symbol, id符号映射词典, 类型dict
        param: symbol, 待加入符号, 类型string
        """
        new_id = len(symbol2id)
        symbol2id[symbol] = new_id
        id2symbol[new_id] = symbol

    def prepare_dict(self):
        """
        根据语料库准备词典
        """
        print("构建字典ing...")
        for token, pos_tag in self.read_lines():
            if not token in self.token2id:
                self._update_dict(self.token2id,
                                  self.id2token,
                                  token)

            if not pos_tag in self.tag2id:
                self._update_dict(self.tag2id,
                                  self.id2tag,
                                  pos_tag
                                  )
        print("字典构建完毕.")
