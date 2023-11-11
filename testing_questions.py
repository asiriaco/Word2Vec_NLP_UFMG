import re
import gensim
from gensim.models import Word2Vec
import pandas as pd


model = Word2Vec.load('model6.model')

queries = []
error_sample = []
	
with open('questions-words.txt') as question_words:
  [queries.append(line.replace("\n", "")) for line in question_words]



def test_query(w1, w2, w3, target, sampling=False):
  predicted = model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
  hit =  1 if predicted[0][0] == target else 0
##  if not hit:
#  	error_sample.append([w1, w2, w3, target, predicted[0][0]])
  return hit
  
queries_test = []
hits = 0
attempts = 0


accuracy_per_class = dict()
current_class = queries[0]

for query in queries[1:]:
  if not re.match("([a-zA-Z]+\s){3}([a-zA-Z])", query):
    accuracy_per_class[current_class.replace(": ", "")] = hits/attempts
    print("class: {}\n accuracy:{}".format(current_class, hits/attempts))
    current_class = query
    hits, attempts = 0, 0

    continue
  try:
    query = list(map(lambda x: x.lower().strip(), query.split(" ")))
    hits += test_query(query[0], query[1], query[2], query[3], sampling=True)
    attempts += 1
  except:
    pass
    
df = pd.DataFrame.from_dict([accuracy_per_class])
df.to_csv("model6_accuracies.csv", index=False)
