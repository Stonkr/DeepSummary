import math
import constants
import numpy as np
from nltk import sent_tokenize
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import nltk

# nltk.download('averaged_perceptron_tagger')


def get_sentence_from_text(text):
    sentence_crude_list = list(sent_tokenize(text.lower()))
    sentence_list = [i.strip() for i in sentence_crude_list]
    return sentence_list


def get_embedder(model_type):
    print(f"Started Loading Model: {model_type} ...")
    return SentenceTransformer(model_type)


def set_embedder(model_type=constants.SUMMARY_MODEL_NAME):
    constants.EMBEDDER = get_embedder(model_type)


def get_embedding_from_sentence(data, normalise=False):
    if isinstance(data, str):
        data = get_sentence_from_text(data)
    if constants.EMBEDDER is None:
        constants.EMBEDDER = get_embedder(constants.SUMMARY_MODEL_NAME)
    sentence_embeddings = constants.EMBEDDER.encode(data)
    if normalise:
        sentence_embeddings_norm = sentence_embeddings / np.linalg.norm(sentence_embeddings,
                                                                        axis=1, keepdims=True)
        return sentence_embeddings_norm
    return sentence_embeddings


def run_pca_on_embedding(embedding, n_components=None, variance_to_explain=0.95):
    if n_components is not None:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA()
    pca_components = pca.fit_transform(embedding)
    try:
        get_most_variation = np.where(np.cumsum(pca.explained_variance_ratio_) > variance_to_explain)[0][0]
    except IndexError:
        get_most_variation = embedding.shape[0]
    pca_components_subset = pca_components[:, 0:get_most_variation]
    return pca_components_subset


def get_optimum_kmeans_cluster_number(embeddings, max_cluster=None):
    # https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-021-01910-w
    if max_cluster is None:
        max_cluster = int(math.sqrt(len(embeddings)))
    inertia_mapping = {}
    max_cluster = max(4, max_cluster)
    for n_cluster in range(1, max_cluster):
        clustering_model = KMeans(n_clusters=n_cluster, random_state=constants.RANDOM_STATE)
        clustering_model.fit(embeddings)
        inertia_mapping[n_cluster] = clustering_model.inertia_ / len(embeddings)
    n_vector = list(inertia_mapping.keys())
    k_vector = list(MinMaxScaler((0, 10)).fit_transform(np.array(
        list(inertia_mapping.values())).reshape(-1, 1)).reshape(len(inertia_mapping), ))
    elbow_points = list(zip(k_vector, n_vector))
    result = []
    for i in range(0, len(elbow_points) - 2):
        p1 = elbow_points[i]
        p2 = elbow_points[i + 1]
        p3 = elbow_points[i + 2]
        a = distance.euclidean(p1, p2)
        b = distance.euclidean(p2, p3)
        c = distance.euclidean(p3, p1)
        result.append(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    return np.argmin(result) + 2


def cluster_kmeans(embeddings, clustering_model=None, n_cluster=3):
    if clustering_model is None:
        clustering_model = KMeans(n_clusters=n_cluster)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_centroids = clustering_model.cluster_centers_
    return cluster_assignment, cluster_centroids