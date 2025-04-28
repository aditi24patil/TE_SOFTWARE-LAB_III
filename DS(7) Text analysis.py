#Extract Sample document and apply following document preprocessing methods: Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
#Create representation of documents by calculating Term Frequency and Inverse DocumentFrequency.

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Sample document
doc1 = "Text analytics is a process of deriving information from text. It includes tasks like tokenization and stemming."
doc2 = "The goal of text analytics is to extract meaningful patterns and knowledge from unstructured text data."
documents = [doc1, doc2]

# -----------------------------
# Preprocessing Step-by-Step
# -----------------------------

# Tokenization
tokens = [word_tokenize(doc.lower()) for doc in documents]
print("\nTokenization:")
print(tokens)

# POS Tagging
pos_tags = [pos_tag(doc) for doc in tokens]
print("\nPOS Tagging (first document):")
print(pos_tags[0])

# Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [[word for word in doc if word not in stop_words and word not in string.punctuation] for doc in tokens]
print("\nAfter Stopword Removal:")
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_docs = [[stemmer.stem(word) for word in doc] for doc in filtered_tokens]
print("\nAfter Stemming:")
print(stemmed_docs)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_docs = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_tokens]
print("\nAfter Lemmatization:")
print(lemmatized_docs)

# -----------------------------
# TF-IDF Representation
# -----------------------------

# Rejoin lemmatized tokens into text format
preprocessed_docs = [' '.join(doc) for doc in lemmatized_docs]

# Apply TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# Display TF-IDF
print("\nTF-IDF Feature Names:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
