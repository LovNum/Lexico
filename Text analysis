from pypdf import PdfReader
import spacy
from collections import Counter
from itertools import combinations
import re
import pandas as pd
import os

# Load PDF
pdf = ""  # Provide the PDF file name or path
reader = PdfReader(pdf)
text = "".join(page.extract_text() for page in reader.pages)

# Save extracted text to a file
with open(f'{pdf[:-4]}.txt', 'w', encoding='utf-8') as file:
    file.write(text)

# Load NLP model and process text
nlp = spacy.load("fr_core_news_lg")
doc = nlp(text)
filtered_tokens = [
    token for token in doc
    if not token.is_stop
    and token.pos_ not in {"NUM", "CCONJ", "SCONJ", "PRON", "ADP", "DET", "PROPN", "AUX", "PART"}
    and not token.is_digit
    and len(token.text) > 2
]
text = " ".join(token.text for token in filtered_tokens)

# Function to extract n-grams
def extract_ngrams(txt, n=2):
    words = re.findall(r'\w+', txt)
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)

trigrams = extract_ngrams(text, n=3)
bigrams = extract_ngrams(text, n=2)
total_words = Counter(re.findall(r'\w+', text))
threshold = 0.3
merge_dict = {}

# Merge n-grams based on occurrence ratio
for ngram, count in trigrams.items():
    first_word = ngram[0]
    if total_words[first_word] > 0 and (count / total_words[first_word]) >= threshold:
        merge_dict[" ".join(ngram)] = "_".join(ngram)

for ngram, count in bigrams.items():
    first_word = ngram[0]
    phrase = " ".join(ngram)
    if phrase not in merge_dict and total_words[first_word] > 0 and (count / total_words[first_word]) >= threshold:
        merge_dict[phrase] = "_".join(ngram)

# Function to replace n-grams in text
def replace_ngrams(txt, merge_dict):
    for phrase, replacement in merge_dict.items():
        txt = re.sub(rf'\b{re.escape(phrase)}\b', replacement, txt, flags=re.IGNORECASE)
    return txt

text = replace_ngrams(text, merge_dict)

doc = nlp(text)
lemmatized_text = " ".join(
    token.text if token.text in merge_dict else token.lemma_
    for token in doc
    if not token.is_digit and not token.is_stop and not token.is_punct and not token.is_space
)
sentences = [sent.text for sent in nlp(lemmatized_text).sents]

# Compute word occurrences
occurrences = Counter()
for sent in sentences:
    tokens = sent.split()
    occurrences.update(tokens)

# Save occurrences to CSV
csv_file = "word_occurrences.csv"
doc_name = pdf[:-4]
df_new = pd.DataFrame(list(occurrences.items()), columns=["Word", doc_name])
df_new[doc_name] = df_new[doc_name].astype(int)

if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    df_final = df_existing.merge(df_new, on="Word", how="outer").fillna(0)
    df_final[doc_name] = df_final[doc_name].astype(int)
    df_final = df_final[["Word"] + [col for col in df_final.columns if col != "Word"]]
else:
    df_final = df_new[["Word", doc_name]]

df_final.to_csv(csv_file, index=False)
print(f"File {csv_file} successfully updated!")

# Topic Modeling
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=1)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    top_n_words=5,
    nr_topics=20
)

topics, probs = topic_model.fit_transform(sentences)
topic_info = topic_model.get_topic_info()

topics_dict = {}
used_words = set()

for topic_id in topic_info["Topic"]:
    if topic_id != -1:
        words = []
        for word, _ in topic_model.get_topic(topic_id):
            if word not in used_words:
                words.append(word)
                used_words.add(word)
        topics_dict[topic_id] = {"default_name": f"Topic {topic_id}", "Words": words}

print("\nTopics:")
for topic_id, topic_data in topics_dict.items():
    topic_name = topic_data["default_name"]
    print(f"\n{topic_name}: {', '.join(topic_data['Words'][:5])}")
    new_name = input(f"Rename '{topic_name}'? (Leave blank to keep default): ")
    if new_name.strip():
        topics_dict[topic_id]["new_name"] = new_name.strip()

print("\n--- Final Topic Summary ---")
for topic_id, topic_data in topics_dict.items():
    topic_name = topic_data.get("new_name", topic_data["default_name"])
    print(f"{topic_name}: {', '.join(topic_data['Words'][:5])}")

# Save topics to CSV
csv_topics = "renamed_topics.csv"
topics_list = []
existing_topics = pd.read_csv(csv_topics) if os.path.exists(csv_topics) else pd.DataFrame(columns=["Topic", "Associated_Words"])

for topic_id, topic_data in topics_dict.items():
    new_name = topic_data.get("new_name", None)
    if new_name:
        new_words = set(topic_data["Words"])
        if new_name in existing_topics["Topic"].values:
            old_words = set(existing_topics.loc[existing_topics["Topic"] == new_name, "Associated_Words"].iloc[0].split(", "))
            merged_words = sorted(new_words.union(old_words))
        else:
            merged_words = sorted(new_words)
        topics_list.append({"Topic": new_name, "Associated_Words": ", ".join(merged_words)})

df_topics = pd.DataFrame(topics_list)
updated_topics = pd.concat([existing_topics, df_topics]).drop_duplicates(subset=["Topic"], keep="last").reset_index(drop=True) if not existing_topics.empty else df_topics
updated_topics.to_csv(csv_topics, index=False)
print(f"File {csv_topics} successfully updated!")
