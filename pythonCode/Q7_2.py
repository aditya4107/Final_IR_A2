import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations

class FeatureVectorLoader:
    def __init__(self, file_path):
        self.feature_vectors = self.load_feature_vectors(file_path)

    def load_feature_vectors(self, file_path):
        feature_vectors = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                docid = parts[0]
                scores = {}
                for item in parts[1:]:
                    word_score_pairs = item.split()
                    for word_score_pair in word_score_pairs:
                        word, score = word_score_pair.split(':')
                        scores[word] = float(score)
                feature_vectors[docid] = scores
        return feature_vectors

    def get_scores(self, docid):
        if docid in self.feature_vectors:
            return list(self.feature_vectors[docid].values())
        else:
            return None

docid_list = []

def retrieve_docids_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)

final_data_file = os.path.join(os.getcwd(),'pythonCode', 'processedData', 'processedData.txt')

if os.path.exists(final_data_file):
    retrieve_docids_from_file(final_data_file)
else:
    print(f"File not found: {final_data_file}")

def load_relevance_data(file_path, docid_list):
    relevance_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                # Initialize relevance scores for all documents to 0
                relevance_data[query_id] = {doc: 0 for doc in docid_list}
            # Update relevance score if available
            relevance_data[query_id][doc_id] = relevance_score
    return relevance_data

trainingQuery_id = []
testQuery_id = []

def read_queries_from_file_training(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            trainingQuery_id.append(query_id)

def read_queries_from_file_test(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            testQuery_id.append(query_id)

processed_queries_folder = os.path.join(os.getcwd(),'pythonCode', 'processedQueries')
files_test = ['test_queries.txt']
files_training = ['training_queries.txt']

for file_name in files_training:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file_training(file_path)
    else:
        print(f"File not found: {file_path}")

for file_name in files_test:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file_test(file_path)
    else:
        print(f"File not found: {file_path}")

pathDocVector = os.path.join('pythonCode', 'output', 'Q7_feature_vectors.txt')
pathTrainingVector = os.path.join('pythonCode', 'output', 'Q7_training_feature_vectors.txt')
pathTestVector = os.path.join('pythonCode', 'output', 'Q7_test_feature_vectors.txt')
relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')

document_loader = FeatureVectorLoader(pathDocVector)
training_loader = FeatureVectorLoader(pathTrainingVector)
test_loader = FeatureVectorLoader(pathTestVector)
relevance_data = load_relevance_data(relevance_file_path, docid_list)

print(len(relevance_data))