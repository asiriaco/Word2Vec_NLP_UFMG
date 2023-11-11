import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
import multiprocessing
import logging  # Setting up the loggings to monitor gensim


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

cores = multiprocessing.cpu_count()

corpus = api.load('text8')

model = Word2Vec(
    sentences=corpus,
    vector_size = 400,
    sample=6e-5,
    alpha=0.03,
    min_alpha=0.0007,
    workers=cores-1,
    min_count = 20,
    window=3
)

model.build_vocab(corpus, progress_per=1000)

model.train(corpus, total_examples=model.corpus_count, epochs=25)

model.save("model4.model")
