from sklearn.feature_extraction.text import CountVectorizer


with open("./nlp_test1.txt") as f:
    corpus = [f.read()]

corpus2=["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]


vectorizer = CountVectorizer(input=corpus2)

text = vectorizer.fit_transform(corpus)
print(text)  # <class 'scipy.sparse.csr.csr_matrix'>
vec_text = text.toarray()
print(vec_text)
features = vectorizer.get_feature_names()
print(features)