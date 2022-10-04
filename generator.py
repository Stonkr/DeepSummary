from helper import *
from scipy.spatial import KDTree
from collections import Counter
import pandas as pd


class SummaryGeneratorExtractive:

    def __init__(self):
        self.summary = ""
        self.df = pd.DataFrame()
        self.sentences = None

    def get_sentence_score(self, optimal_cluster_size, embeddings, cluster_assignment, cluster_centroids):
        master_order = []
        for i in range(0, optimal_cluster_size):
            embeddings_in_cluster = embeddings[np.where(cluster_assignment == i)]
            sentences_in_cluster = self.sentences[np.where(cluster_assignment == i)]
            tree = KDTree(embeddings_in_cluster)
            centroid = cluster_centroids[i]
            sentence_order = tree.query(centroid, len(embeddings_in_cluster))[1]
            for j, sentence in enumerate(sentences_in_cluster):
                rating = np.where(sentence_order == j)[0][0] / (len(sentences_in_cluster) - 1)
                if rating <= 1:
                    pass
                else:
                    rating = 0
                master_order.append(rating)
        return master_order

    def get_summary_result(self, text, **kwargs):
        try:
            ratio = kwargs.get("ratio", 0.4)
            self.sentences = np.array(get_sentence_from_text(text))
            total_sentences = len(self.sentences)
            self.df = pd.DataFrame(self.sentences, columns=["sentences"])

            embeddings = get_embedding_from_sentence(self.sentences, None)
            pca_components = run_pca_on_embedding(embeddings)
            optimum_clusters = get_optimum_kmeans_cluster_number(pca_components)
            cluster_assignment, cluster_centroids = cluster_kmeans(pca_components,
                                                                   n_cluster=optimum_clusters)
            self.df["cluster"] = cluster_assignment
            self.df["score"] = self.get_sentence_score(optimum_clusters, pca_components, cluster_assignment,
                                                       cluster_centroids)

            cluster_ratio = dict(Counter(cluster_assignment))
            total_parts = sum(cluster_ratio.values())
            sentence_to_get_per_cluster = dict(zip(cluster_ratio.keys(),
                                                   [int(((total_sentences * ratio) / total_parts) * i)
                                                    for i in cluster_ratio.values()]))

            combined_df = []
            for cluster in sentence_to_get_per_cluster:
                sentence_to_filter = sentence_to_get_per_cluster[cluster]
                filter_df = self.df[self.df["cluster"] == cluster].sort_values("score")
                combined_df.append(filter_df[:sentence_to_filter])
            summary_df = pd.concat(combined_df)
            summary_sentences = list(summary_df.sort_index()["sentences"])

            # if self.sentences[0] not in summary_sentences:
            #     summary_sentences.insert(0, self.sentences[0])
            # if self.sentences[-1] not in summary_sentences:
            #     summary_sentences.append(self.sentences[-1])

            summary_sentences = [summary_sentence[0].upper() + summary_sentence[1:] for summary_sentence in
                                 summary_sentences]
            summary = " ".join(summary_sentences)
        except Exception as e:
            print(e)
            summary = text

        return summary