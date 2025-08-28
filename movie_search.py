import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger()


# Load dataset and create embeddings (global for testing)
def load_data(path:str)->pd.DataFrame:
    data=pd.read_csv(path)
    logger.debug("Data Loaded")
    return data

# Load the Sentence Transformer model
def load_model(name:str="all-MiniLM-L6-v2")->SentenceTransformer:
    model = SentenceTransformer(name)
    logger.debug("Model Loaded")
    return model
# Convert the 'plot of the movies into an embedding
def plot_to_embeddings(plot:str,model:SentenceTransformer)->List:
    return model.encode([plot], convert_to_tensor=False)

def search_movies(query:str,top_n:int=3)->pd.DataFrame:
    data=load_data("movies.csv")
    model=load_model()
    search_query_embedding=plot_to_embeddings(query,model)
    data['similarity']=data['plot'].apply(lambda x:cosine_similarity(plot_to_embeddings(x,model),search_query_embedding)[0][0])
    logger.debug("Similarity Computed")
    return data.sort_values(by=['similarity'],ascending=False).reset_index().drop(columns=['index']).head(top_n)

    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <query>")
    else:
        query = sys.argv[1]
        print(search_movies(query))
    