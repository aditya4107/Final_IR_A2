import os
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LinearRegression

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

    def get_word_scores(self, docid):
        if docid in self.feature_vectors:
            return {word: score for word, score in self.feature_vectors[docid].items()}
        else:
            return None

def retrieve_docids_from_file(file_path):
    docid_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)
    return docid_list

def load_relevance_data(file_path, docid_list):
    relevance_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = {doc: 0 for doc in docid_list}
            relevance_data[query_id][doc_id] = relevance_score
    return relevance_data

def read_queries_from_file_training(file_path):
    trainingQuery_id = []
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            trainingQuery_id.append(query_id)
    return trainingQuery_id

def read_queries_from_file_test(file_path):
    testQuery_id = []
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            testQuery_id.append(query_id)
    return testQuery_id


class PairwiseRanking:
    def __init__(self, document_loader, relevance_data):
        self.document_loader = document_loader
        self.relevance_data = relevance_data
        self.model = None

    def calculate_relevance_score(self, doc1, doc2, query_id):
        scores1 = self.document_loader.get_scores(doc1)
        scores2 = self.document_loader.get_scores(doc2)
        relevance_scores = self.relevance_data[query_id]
        relevance_doc1 = relevance_scores.get(doc1, 0)
        relevance_doc2 = relevance_scores.get(doc2, 0)
        return relevance_doc1 - relevance_doc2

    def generate_pairs(self, query_id):
        doc_ids = list(self.relevance_data[query_id].keys())
        pairs = list(combinations(doc_ids, 2))
        return pairs

    def train(self, training_queries, k):
        X = []
        y = []
        count = 0
        i = 0
        for query_id in training_queries:
            print(i)
            i += 1
            if count >= k:
                break
            pairs = self.generate_pairs(query_id)
            for pair in pairs:
                doc1, doc2 = pair
                score = self.calculate_relevance_score(doc1, doc2, query_id)
                X.append([score])
                y.append(1 if score > 0 else -1)
            count += 1
        X = np.array(X)
        y = np.array(y)
        self.model = LinearRegression().fit(X, y)

    def rank_documents(self, test_queries):
        results = defaultdict(list)
        for query_id in test_queries:
            doc_ids = list(self.relevance_data[query_id].keys())
            X = []
            for doc_id in doc_ids:
                score = self.calculate_relevance_score(doc_id, '', query_id)
                X.append([score])
            X = np.array(X)
            predicted_scores = self.model.predict(X)
            sorted_docs = [doc for _, doc in sorted(zip(predicted_scores, doc_ids), reverse=True)]
            results[query_id] = sorted_docs[:10]
        return results


def write_results_to_file(results, output_file):
    with open(output_file, 'w') as file:
        for query_id, docs in results.items():
            file.write(f"{query_id}\t{' '.join(docs)}\n")

# Load data
final_data_file = os.path.join(os.getcwd(), 'pythonCode', 'processedData', 'processedData.txt')
pathDocVector = os.path.join('pythonCode', 'output', 'Q7_feature_vectors.txt')
pathTrainingVector = os.path.join('pythonCode', 'output', 'Q7_training_feature_vectors.txt')
pathTestVector = os.path.join('pythonCode', 'output', 'Q7_test_feature_vectors.txt')
relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')

docid_list = retrieve_docids_from_file(final_data_file)
document_loader = FeatureVectorLoader(pathDocVector)
training_loader = FeatureVectorLoader(pathTrainingVector)
test_loader = FeatureVectorLoader(pathTestVector)
relevance_data = load_relevance_data(relevance_file_path, docid_list)
trainingQuery_id = read_queries_from_file_training(os.path.join('pythonCode', 'processedQueries', 'training_queries.txt'))
testQuery_id = read_queries_from_file_test(os.path.join('pythonCode', 'processedQueries', 'test_queries.txt'))

# Initialize pairwise ranking model
pairwise_ranking = PairwiseRanking(document_loader, relevance_data)

# Train model
pairwise_ranking.train(trainingQuery_id,10)

# Rank documents for test queries
results = pairwise_ranking.rank_documents(testQuery_id)

# Write results to file
output_file = os.path.join('pythonCode', 'output', 'Q7_2_results.txt')
write_results_to_file(results, output_file)
