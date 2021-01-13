from models.hmm import HMMPOSTagger
from utils.corpus import CorpusHelper

def test(tagger, sentence):
    pos_tags = tagger.decode(sentence)
    print("sentence: ", sentence)
    print("tags: ",' '.join(pos_tags))


if __name__ == '__main__':
    corpus_helper = CorpusHelper("./data/traindata.txt")
    print("Number of tags: {}\nNumber of tokens: {}".format(len(corpus_helper.tag2id), len(corpus_helper.token2id)))
    print(corpus_helper.tag2id)

    tagger = HMMPOSTagger(corpus_helper)
    tagger.train()

    test(tagger, "You are a good teacher .")
    test(tagger, "You are a bad man .")
    test(tagger, "She is a pretty girl .")
    test(tagger, "He likes games .")
    test(tagger, "I knew him before .")


