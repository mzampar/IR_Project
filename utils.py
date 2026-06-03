
# **********************************************************
# utils.py
#
# Utilities to process text, normalise covid terms, 
# compute ndcg (normalised discounted cumulative gain)
#
# **********************************************************

# --- MODULES ---

import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np

# Ensure the necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# --- FUNCTIONS ---

def normalize_covid_terms(text):

    text = re.sub(r'\b(alpha|beta|gamma|delta)[-\s]?(corona.?virus(es)?|covid.?19|covid.?2019|sars.?cov.?2)\b', 'covid19', text)
    text = re.sub(r'\bcorona.?virus(es)?\b', 'covid19', text)
    text = re.sub(r'\bcovid.?19\b', 'covid19', text)
    text = re.sub(r'\bcovid.?2019\b', 'covid19', text)
    text = re.sub(r'\bsars.?cov.?2\b', 'covid19', text)
    text = re.sub(r'\bwuhan.?virus\b', 'covid19', text)
    text = re.sub(r'\b2019.?n.?cov\b', 'covid19', text)
    text = re.sub(r'\b2019_ncov\b', 'covid19', text)

    # Remove possible spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text(text, remove_covid=False):
    # Convert to lowercase
    text = text.lower()

    # Remove accents/diacritics
    text = unidecode(text)

    # Normalise covid terms
    text = normalize_covid_terms(text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove numbers
    tokens = [
        t for t in tokens
        if not re.fullmatch(r'\d+(\.\d+)?', t)
    ]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    #tokens = [lemmatizer.lemmatize(token) for token in tokens]
    pos_tags = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]

    if remove_covid:
        tokens = [t for t in tokens if t != "covid19"]

    text = ' '.join(tokens)

    return text

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def compute_ndcg(
    retrieved_doc_ids,
    rel_ids,
    rel_scores,
    normalize=True,
    n=None,
):
    """
    Compute nDCG for a list of retrieved documents 
    Assuming retrieved_doc_ids is sorted from most relevant to less relevant

    Parameters:
    - retrieved_doc_ids: List of retrieved document IDs.
    - rel_ids: List of relevant document IDs.
    - rel_scores: List of relevance scores for the relevant documents.
    - normalize: If True, normalize by ideal DCG.
    - n: Cutoff (nDCG@k). If None, use all retrieved.

    Returns:
    - ndcg: Normalized DCG score.
    - dcg: Discounted cumulative gain.
    - idcg: Ideal discounted cumulative gain.
    """

    # Join information about corpus-id and relevance score
    rel_dict = dict(zip(rel_ids, rel_scores))

    # Keep first k retrieved documents
    if n is not None:
        retrieved_doc_ids = retrieved_doc_ids[:n]

    # Get relevance scores
    gains = []
    for doc_id in retrieved_doc_ids:
        # Get 1 or 2 if relevant, 0 if not present in qrels (not relevant)
        rel = rel_dict.get(doc_id, 0.0)
        #gain = (2 ** rel) - 1
        gain = rel
        gains.append(gain)

    # Compute DCG
    dcg = 0.0
    # Assuming retrieved_doc_ids is sorted from most relevant to less relevant
    for i, gain in enumerate(gains):
        rank = i + 1
        discount = np.log2(rank + 1)
        dcg += gain / discount

    # Compute IDCG
    # Sort true relevances descending
    ideal_rels = sorted(rel_scores, reverse=True)

    # Keep first k relevant documents
    if n is not None:
        ideal_rels = ideal_rels[:len(retrieved_doc_ids)]

    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        # gain =(2 ** rel) - 1
        gain = rel
        rank = i + 1
        discount = np.log2(rank + 1)
        idcg += gain / discount

    # Final nDCG
    ndcg = dcg / idcg if normalize and idcg > 0 else dcg

    return ndcg, dcg, idcg


