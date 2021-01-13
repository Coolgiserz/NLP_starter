import numpy as np
from scipy.special import logsumexp
from scipy.sparse import lil_matrix
import sys
from utils.corpus import CorpusHelper
class HMMPOSTagger:
    """
    HMM 词性标注模型，实现模型的定义，训练和预测等功能
    HMM 参数:
        初始状态概率向量 pi,
        状态转移概率矩阵 A,
        观测概率矩阵    B
    """

    def __init__(self, corpus_helper, eps=None):
        """
        param: corpus_helper，语料库辅助类实例，类别CorpusHelper
        param: eps, 极小值，用于平滑log计算，类别float
        """
        self.corpus_helper = corpus_helper
        self.n_tokens = len(corpus_helper.token2id)
        self.n_tags = len(corpus_helper.tag2id)
        self.pi = np.zeros(self.n_tags, dtype=np.float)
        self.A = np.zeros((self.n_tags, self.n_tags), dtype=np.float)
        self.B = np.zeros((self.n_tags, self.n_tokens), dtype=np.float)
        self.eps = np.finfo(float).eps if eps is None else eps

    def train(self):
        """
        训练模型，完成语料库的统计工作
        """
        print("开始训练")
        last_tag_id = None  # 记录前一个tag，若其值为None则表明当前为新句开始。
        for token_id, tag_id in self.corpus_helper.read_lines2id():

            # 无论如何都要更新B的统计
            self.B[tag_id, token_id] += 1

            if last_tag_id is None:
                # 若当前是新句子的开始，需要更新pi
                self.pi[tag_id] += 1
            else:
                # 否则，更新A
                self.A[last_tag_id, tag_id] += 1

            # 更新上一时刻tag
            last_tag_id = None if self.corpus_helper.is_end_tokenid(token_id) else tag_id

        # 转化为概率
        self.pi = self.pi / np.sum(self.pi)
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        self.B = self.B / np.sum(self.B, axis=1, keepdims=True)

        print("训练结束")
        print("pi:{}".format(self.pi))
        print("A[0,:]:\n{}".format(self.A[0]))

    def _log(self, p):
        """
        log 函数，考虑平滑
        """
        return np.log(p + self.eps)

    def decode(self, sentence):
        """
        给定句子，使用Viterbi算法找到最佳词性标注序列
        param: sentence, 输入句子, 类型string
        return:词性标注序列, 类型list[string]
        """
        if not sentence:
            print("请输入句子")
            return ""

        # (这里没有考虑未登录词的情况)
        token_ids = [self.corpus_helper.token2id[token] for token in sentence.split(" ")]
        n_tags, n_tokens = self.n_tags, len(token_ids)
        A, B = self.A, self.B

        # 初始化动态规划存储矩阵和记录最佳路径的回溯矩阵
        dp = np.zeros((n_tags, n_tokens), dtype=np.float)
        traces = np.zeros((n_tags, n_tokens), dtype=np.int)

        # 初始化第一个token的位置
        for i in range(n_tags):
            dp[i, 0] = self._log(self.pi[i]) + self._log(self.B[i, token_ids[0]])

        # 动态规划更新第二个token开始的分数
        for t in range(1, n_tokens):

            token_id = token_ids[t]  # 当前token id

            for i in range(n_tags):

                dp[i, t] = -sys.maxsize  # 初始值为系统最小值

                for k in range(n_tags):
                    score = dp[k, t - 1] + self._log(A[k, i]) + self._log(B[i, token_id])

                    if score > dp[i, t]:
                        dp[i, t] = score
                        traces[i, t] = k

        # dp中最佳路径的最终tag
        last_best_tag = np.argmax(dp[:, -1])

        # 回溯最佳路径
        decoded = [0] * n_tokens

        decoded[-1] = last_best_tag
        for t in range(n_tokens - 1, 0, -1):
            last_best_tag = traces[last_best_tag, t]
            decoded[t - 1] = last_best_tag

        pos_tags = self.corpus_helper.id_to_tags(decoded)
        return pos_tags