from pypdf import PdfReader
import re
from collections import Counter
from itertools import combinations
import spacy
import networkx as nx

# Dictionary of themes with associated words
themes = {
    "theme": ["word_1", "word_2"]

}

pdf_files = []

# Extract text from PDFs
text = ""
for pdf in pdf_files:
    reader = PdfReader(pdf)
    for page in reader.pages:
        text += page.extract_text() + " "

# Initialize SpaCy model
nlp = spacy.load("fr_core_news_lg")
nlp.max_length = len(text) + 1000000

doc = nlp(text)
tokens_filtres = [
    token for token in doc
    if not token.is_stop
    and token.pos_ != "NUM"     
    and token.pos_ != "CCONJ" 
    and token.pos_ != "SCONJ" 
    and token.pos_ != "PRON"
    and token.pos_ != "ADP" 
    and token.pos_ != "DET" 
    and token.pos_ != "PROPN" 
    and token.pos_ != "AUX" 
    and token.pos_ != "PART" 
    and not token.is_digit
    
    and len(token.text) > 2
]
text = " ".join(token.text for token in tokens_filtered)

# Extract n-grams
def extract_ngrams(txt, n=2):
    words = re.findall(r'\w+', txt)
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ngrams)

trigrams = extract_ngrams(text, n=3)
bigrams = extract_ngrams(text, n=2)
word_count = Counter(re.findall(r'\w+', text))

# Create theme-based co-occurrence graph
G = nx.Graph()
for theme in themes.keys():
    G.add_node(theme, label=theme)

for (theme1, theme2), count in Counter(combinations(themes.keys(), 2)).items():
    G.add_edge(theme1, theme2, weight=count)

print(f"Total edges (theme connections): {G.number_of_edges()}")
nx.write_gexf(G, "theme_cooccurrences.gexf")

