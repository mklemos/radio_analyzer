from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def segment_topics(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    clusters = model.labels_.tolist()
    terms = vectorizer.get_feature_names_out()
    return clusters, terms
