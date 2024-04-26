import os
from collections import defaultdict

def retrieve_docids_from_file(file_path):
    docid_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)
    return docid_list

def load_relevance_data(file_path, docid_list):
    relevance_data = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = {}
            relevance_data[query_id][doc_id] = relevance_score
    return relevance_data

def normalize_scores(scores):
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {doc_id: 0 for doc_id in scores}
    normalized_scores = {}
    for doc_id, score in scores.items():
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_scores[doc_id] = normalized_score
    return normalized_scores

def calculate_ndcg(query_id, my_ranking, relevance_data, k):
    ideal_ranking = relevance_data.get(query_id, {})
    if not ideal_ranking:
        return 0  # If no ideal ranking is available for the query, NDCG is 0.
    
    # Take top k documents from the ideal ranking
    ideal_sorted = sorted(ideal_ranking.items(), key=lambda x: (-x[1], x[0]))
    ideal_top_k = ideal_sorted[:k]
    if len(ideal_top_k) < k:
        return None  # Not enough relevant documents for NDCG calculation
    
    ideal_top_k_docs = [doc_id for doc_id, _ in ideal_top_k]

    # Find corresponding documents in myRanking
    my_top_k_scores = {doc_id: my_ranking.get(doc_id, 0) for doc_id in ideal_top_k_docs}

    # Normalize scores
    ideal_normalized_scores = normalize_scores(ideal_ranking)
    my_normalized_scores = normalize_scores(my_top_k_scores)

    # Calculate DCG and IDCG
    dcg = sum((ideal_normalized_scores[doc_id] / (i + 1)) for i, doc_id in enumerate(ideal_top_k_docs))
    idcg = sum((ideal_normalized_scores[doc_id] / (i + 1)) for i, doc_id in enumerate(sorted(ideal_normalized_scores.keys())))
    
    # Calculate NDCG
    if idcg == 0:
        return 0
    ndcg = dcg / idcg
    return ndcg

def calculate_ndcg_for_ranking(myRanking, k, query_id):
    final_data_file = os.path.join(os.getcwd(), 'pythonCode', 'processedData', 'processedData.txt')
    docid_list = retrieve_docids_from_file(final_data_file)
    relevance_folder = os.path.join('relevance')
    relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')
    relevance_data = load_relevance_data(relevance_file_path, docid_list)

    # Calculate NDCG using the provided ranking (myRanking)
    ndcg_score = calculate_ndcg(query_id, myRanking, relevance_data, k)
    if ndcg_score is None:
        print("Not enough relevant documents for NDCG calculation.")
    else:
        print("NDCG Score:", ndcg_score)
