import re
import csv
from collections import defaultdict
import numpy as np
import os

# Load the Knowledge Graph (GENA) from the CSV file
knowledge_graph = defaultdict(list)
with open('gena_data_final_triples.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        subject, relation, obj = row
        knowledge_graph[subject].append((relation, obj))
        knowledge_graph[obj].append((relation, subject))

# Loading queries
queryList = []

def read_queries_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queryList.append((query_id, query_text))

files = ['/content/dev_queries.txt', '/content/test_queries.txt', '/content/training_queries.txt']

for file_name in files:
    read_queries_from_file(file_name)

# Loading documents
documents = {}

def load_documents_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid, text = line.strip().split('\t', 1)
            documents[docid] = text

load_documents_from_file('/content/processedData.txt')

# Load stop words from the file
with open('/content/stopwords.large', 'r') as f:
    stop_words = set(line.strip() for line in f)


import spacy
from collections import Counter

# Load the pre-trained SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    # Create a SpaCy Doc object
    doc = nlp(text)
    
    # Extract all nouns as entities
    entities = [token.text for token in doc if token.pos_ == "NOUN"]
    
    return entities

def bag_of_entities(text):
    entities = extract_entities(text)
    return Counter(entities)

def coordinate_match(query_entities, doc_entities):
    common_entities = query_entities & doc_entities
    query_score = sum(query_entities.values())
    doc_score = sum(doc_entities.values())
    
    if query_score == 0 and doc_score == 0:
        return 0  # Return 0 if both query and document entities are empty
    else:
        return sum(common_entities.values()) / (query_score + doc_score - sum(common_entities.values()))

def entity_frequency_score(query_entities, doc_entities):
    return sum((query_entities & doc_entities).values()) / max(sum(query_entities.values()), sum(doc_entities.values()))

def compute_similarity(query_entities, doc_entities):
    coord_match_score = coordinate_match(query_entities, doc_entities)
    entity_freq_score = entity_frequency_score(query_entities, doc_entities)
    return (coord_match_score + entity_freq_score) / 2


def retrieve_documents(query, documents):
    query_entities = bag_of_entities(query)
    scores = []

    for doc_id, doc_text in documents.items():
        doc_entities = bag_of_entities(doc_text)
        similarity_score = compute_similarity(query_entities, doc_entities)
        scores.append((doc_id, similarity_score))

    # Sort the documents by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores

def retrieve_and_print_documents(queryList, documents, num_queries=10):
    for i, query in enumerate(queryList[:num_queries]):
        query_text = query[1] if isinstance(query, tuple) else query
        query_entities = extract_entities(query_text)

        print(f"Query: {query_text}")
        print(f"Query Entities: {query_entities}")

        scores = retrieve_documents(query_text, documents)

        if not scores:
            print("No relevant documents found.")
        else:
            print("Top 5 relevant documents:")
            for doc_id, similarity_score in scores[:5]:
                doc_text = documents[doc_id]
                print(f"Document ID: {doc_id}, Score: {similarity_score:.4f}")
                print(f"Document Text: {doc_text}")
                print()

        if i < num_queries - 1:
            print("-" * 80)
