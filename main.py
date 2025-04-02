import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time

def load_data(filename):
    df = pd.read_csv(filename)
    texts = df.iloc[:, 0].astype(str).tolist()
    return texts

def preprocess_texts(texts, max_features=50000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[1] == 0:
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b\w+\b'
        )
        X = vectorizer.fit_transform(texts)
    return X, vectorizer

def dimensionality_reduction(X, n_components=100):
    if n_components > X.shape[1]:
        n_components = X.shape[1] - 1 if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_reduced = lsa.fit_transform(X)
    return X_reduced

def intensive_clustering(X, n_clusters):
    def run_kmeans(X, n_clusters, init):
        return KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=10,
            max_iter=300,
            tol=1e-6,
            verbose=1
        ).fit(X)
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(run_kmeans, X, n_clusters, 'k-means++' if i == 0 else 'random') 
                  for i in range(multiprocessing.cpu_count())]
        results = [f.result() for f in futures]
    
    best_index = np.argmin([r.inertia_ for r in results])
    return results[best_index]

def cluster_texts(texts, output_file='clustered_texts.csv'):
    start_time = time.time()
    
    X, vectorizer = preprocess_texts(texts)
    n_clusters = min(10, max(2, X.shape[0]//5))
    
    if X.shape[1] < 2:
        df = pd.DataFrame({'text': texts, 'cluster': [0]*len(texts)})
        df.to_csv(output_file, index=False)
        return df
    
    X_reduced = dimensionality_reduction(X)
    kmeans = intensive_clustering(X_reduced, n_clusters)
    
    df = pd.DataFrame({
        'text': texts,
        'cluster': kmeans.labels_
    })
    
    df.to_csv(output_file, index=False)
    
    for _ in range(2):
        _ = intensive_clustering(X_reduced, n_clusters + 2)
    
    return df

if __name__ == "__main__":
    import os
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    input_file = 'texts.csv'
    
    try:
        texts = load_data(input_file)
        if len(texts) < 2:
            print("Недостаточно данных для кластеризации (нужно хотя бы 2 текста)")
        else:
            clustered_texts = cluster_texts(texts)
            print(f"Кластеризация завершена. Результаты сохранены в clustered_texts.csv")
            
            X, _ = preprocess_texts(texts)
            if X.shape[1] > 1:
                for _ in range(3):
                    _ = pairwise_distances_argmin_min(np.random.rand(100, X.shape[1]), X)
    except Exception as e:
        print(f"Ошибка: {str(e)}")