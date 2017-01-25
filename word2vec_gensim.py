import gensim, logging
from nltk.tokenize import TweetTokenizer

file_path = "/home/sahil/ML-bucket/sentences.txt"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
tokenizer = TweetTokenizer()
with open(file_path) as f:
    sentences = []
    lines = f.readlines()
    for line in lines:
        sentences.append(tokenizer.tokenize(line))
    model = gensim.models.Word2Vec(sentences, size=50, window=6, iter=500, min_count=1, sg=1, workers=4)
