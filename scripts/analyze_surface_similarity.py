from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from tqdm import tqdm
import random

random.seed(23103)

text_data = []
clusters_data = json.load(open("./t0_cluster_data.json"))
for cluster_id, cluster_data in clusters_data.items():
    text = set([x["input"] for x in cluster_data])
    text_data.extend([(x, cluster_id) for x in text])


vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', stop_words='english')
index = vectorizer.fit_transform([x[0] for x in text_data])

closest_in_cluster = []

indices_set = list(range(len(text_data)))
random.shuffle(indices_set)
eval_indices_set = indices_set[:10000]

for i in tqdm(eval_indices_set):
    _, input_cluster = text_data[i]
    query_vector = index[i]
    similarities = cosine_similarity(index, query_vector).flatten()
    # -2 because -1 will be the same as the query.
    most_similar_index = np.argsort(similarities, axis=0)[-2]
    closest_data_point, its_cluster_id = text_data[most_similar_index]
    closest_in_cluster.append(input_cluster == its_cluster_id)


print(f"Closest in cluster: {sum(closest_in_cluster) / len(closest_in_cluster) * 100}%")
