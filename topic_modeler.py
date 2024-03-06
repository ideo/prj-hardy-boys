"""
Uses UMAP to reduce and cluster OpenAI embeddings of the sample data.
"""

import os
import re

import pandas as pd
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from stqdm import stqdm as tqdm

import umap
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from directories import DATA_DIR, EMBEDDINGS_DIR


load_dotenv()


CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SMALL_MODEL = "text-embedding-3-small"
LARGE_MODEL = "text-embedding-3-large"


class TFIDF_Topic_Modeler:
    def __init__(self, tfidf_filename):
        self.tfidf_filepath = EMBEDDINGS_DIR / tfidf_filename

    ### Embeddings

    def attempt_to_find_topics(self, df, columns=["title", "text"]):
        if not os.path.exists(self.tfidf_filepath):
            vectors = self.vectorize(df, columns=columns)
            nmf_reduction = self.reduce_dimensions(vectors)
            data = self.reduce_to_2d(nmf_reduction)

            tfidf_embeddings = pd.DataFrame(data, index=df["post link"])
            tfidf_embeddings.to_pickle(self.tfidf_filepath)
                
        else:
            tfidf_embeddings = pd.read_pickle(self.tfidf_filepath)
        
        return tfidf_embeddings


    def vectorize(self, df, columns=["title", "text"]):
        """
        Clean text and create the TF-IDF vectors
        """
        cleaned_text = self._clean_text(df, columns=columns)
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(cleaned_text)
        return vectors
    

    def reduce_dimensions(self, vectors, n_components=10):
        model = NMF(n_components=n_components)
        reduction = model.fit_transform(vectors)
        return reduction
    

    def reduce_to_2d(self, reduction):
        umap_reducer = umap.UMAP(random_state=42, n_jobs=1)
        futher_reduction = umap_reducer.fit_transform(reduction)
        return futher_reduction


    def _clean_text(self, df, columns=["title", "text"]):
        """
        Preprocess text to prep it for TFIDF vectorizing
        """
        text = df[columns].apply(lambda row: " ".join(row), axis=1).values
        text = [self._remove_newlines_and_whitespace(txt) for txt in text]
        return text


    @staticmethod
    def _remove_newlines_and_whitespace(text):
        pattern = "\n"
        cleaned_text = re.sub(pattern, " ", text)

        pattern = "[ \t]{2,}"
        cleaned_text = re.sub(pattern, " ", cleaned_text)

        pattern = "[ \t]{2,}"
        cleaned_text = re.sub(pattern, " ", cleaned_text)

        return cleaned_text



class Topic_Modeler:
    def __init__(self, embeddings=None, source_data=None):
        # self.embeddings_filepath = embeddings_filepath
        # self.source_data_filepath = source_data_filepath
        # self.embeddings = pd.read_pickle(embeddings_filepath)
        # self.source_data = pd.read_csv(source_data_filepath)
        self.embeddings = embeddings
        self.source_data = source_data


    def reduce_dimensions(self):
        self.umap_model = umap.UMAP(random_state=42, n_jobs=1)
        reduction = self.umap_model.fit_transform(self.embeddings.values)
        return reduction
    

    def cluster(self, reduction, n_clusters):
        self.cluster_model = SpectralClustering(n_clusters=n_clusters)
        self.cluster_model.fit(reduction)
        return self.cluster_model.labels_
    


def get_embeddings(df):
    tqdm.pandas(desc=f"Getting OpenAI Embeddings for {df.shape[0]} posts")

    df.set_index("post link", inplace=True)
    data = (df["title"] + "\n\n" + df["text"]).progress_apply(fetch_embeddings_api_call)

    embeddings = pd.DataFrame(data.values, index=df.index)
    embeddings = embeddings[0].apply(pd.Series)
    return embeddings


def fetch_embeddings_api_call(source_text):
    try:
        response = CLIENT.embeddings.create(
            input=source_text,
            model=SMALL_MODEL,
        )
        embedding = response.data[0].embedding
        return embedding
    except BadRequestError:
        return None


if __name__ == "__main__":
    filepath = "data/hardie+install.csv"
    df = pd.read_csv(filepath)

    get_embeddings(df)