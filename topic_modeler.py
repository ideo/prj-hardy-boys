"""
Uses UMAP to reduce and cluster OpenAI embeddings of the sample data.
"""

import os

import pandas as pd
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from stqdm import stqdm as tqdm
import umap
from sklearn.cluster import SpectralClustering

from directories import DATA_DIR


load_dotenv()


CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SMALL_MODEL = "text-embedding-3-small"
LARGE_MODEL = "text-embedding-3-large"


class Topic_Modeler:
    def __init__(self, embeddings, source_data):
        # self.embeddings_filepath = embeddings_filepath
        # self.source_data_filepath = source_data_filepath
        # self.embeddings = pd.read_pickle(embeddings_filepath)
        # self.source_data = pd.read_csv(source_data_filepath)
        self.embeddings = embeddings
        self.source_data = source_data


    def reduce_dimensions(self):
        self.umap_model = umap.UMAP(random_state=42)
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