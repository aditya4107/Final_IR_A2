from utilityFunctions import retrieve_document_vector_values as getDocVector
import pandas as pd
import os
import math
from math import log, log2
from collections import defaultdict


# taking choice from user
print("For document: enter 1 for nnn, 2 for ntn, 3 for ntc")
choice_doc = input("Enter your choice: ")

print("For query: enter 1 for nnn, 2 for ntn, 3 for ntc")
choice_query = input("Enter your choice: ")


# loading queries
queryList = []

def read_queries_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queryList.append((query_id, query_text))

processed_queries_folder = os.path.join(os.getcwd(),'pythonCode', 'processedQueries')
files = ['dev_queries.txt', 'test_queries.txt', 'training_queries.txt']

for file_name in files:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file(file_path)
    else:
        print(f"File not found: {file_path}")

# df = pd.DataFrame(queryList, columns=['Query ID', 'Query Text'])
# print(df.head())

# loading the list of docids
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


# def dot_product(vector1, vector2):
#     result = 0.0
#     for key in vector1:
#         result += vector1[key] * vector2.get(key, 0.0)
#     return result

# def get_word_value(vector, word):
#     return vector.get(word, 0.0)



# given a test, give its words
def extract_words_from_text(text):
    words = text.split()
    return words

# it gives df and corresponding docids
def load_index_combined(index_combined_file):
    index_combined = defaultdict(lambda: [0, set()])
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, df, doc_ids = line.strip().split('\t')
            index_combined[word][0] = int(df)
            index_combined[word][1] = set(doc_ids.split())
    return index_combined

# it gives word and corresponding docids
def build_index_map(index_file):
    index_map = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word, _, doc_ids_str = parts[:3]
                doc_ids = set(doc_ids_str.split())
                if word not in index_map:
                    index_map[word] = set()
                index_map[word].update(doc_ids)
    return index_map

# searches whether a word exists in a doc
def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0

# get the doc value for a vector
def getdocValue(word):
    if choice_doc == '1':
        return 1.0
    else:
        df =  index_combined[word][0]
        return log(5371 / df)
    
#  get the query value for a vector
def getqueryValue(word):
    if choice_query == '1':
        return 1.0
    else: 
        df = index_combined[word][0]
        return log(5371 / df)

# does normalisation
def cosine_normalization_term(vector):
    squared_sum = sum(x ** 2 for x in vector)
    normalization_term = math.sqrt(squared_sum)
    return normalization_term

# ndcg values
def calculate_ndcg_score(ranking, standard_ranking, k):
    # Sort rankings by relevance score, then by docid
    ranking = sorted(ranking, key=lambda x: (x[1], x[0]), reverse=True)[:k]
    standard_ranking = sorted(standard_ranking, key=lambda x: (x[1], x[0]), reverse=True)[:k]
    
    # Compute DCG for the given ranking
    def compute_dcg(ranking):
        dcg = 0
        for i, (_, relevance) in enumerate(ranking, start=1):
            dcg += (2**relevance - 1) / log2(i + 1)
        return dcg
    
    # Normalize DCG
    def normalize_dcg(dcg, ranking):
        ideal_ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        ideal_dcg = compute_dcg(ideal_ranking)
        return dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    # Calculate NDCG
    dcg = compute_dcg(ranking)
    standard_dcg = compute_dcg(standard_ranking)
    ndcg = normalize_dcg(dcg, ranking) / normalize_dcg(standard_dcg, standard_ranking) if standard_dcg > 0 else 0
    return ndcg

# generates standard ranking
def getStandardRanking(query_id):
    return {}

# Load the index_combined and index_map
current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
index_combined = load_index_combined(index_combined_file)
index_map = build_index_map(index_combined_file)

# main task starts
for query in queryList:
    query_id, query_text = query
    ranking = {}
    for docid in docid_list:
        word_list = extract_words_from_text(query_text)
        similarity = 0.0
        doclist = []
        querylist = []
        for queryWord in word_list:
            value = check_word_in_document(queryWord, docid, index_map)
            if value != 0:
                docval = getdocValue(queryWord)
                queryval = getqueryValue(queryWord)
                doclist.append(docval)
                querylist.append(queryval)
                similarity += value * docval * queryval
        if choice_doc == '3':
            doc_norm = cosine_normalization_term(doclist)
            if doc_norm != 0:
                similarity = similarity / doc_norm
        if choice_query == '3':
            query_norm = cosine_normalization_term(querylist)
            if query_norm!= 0:
                similarity = similarity / query_norm
        ranking[docid] = similarity

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    print(query_id)
    print(query_text)
    for doc_id, score in sorted_ranking[:10]:
        print(f"Document ID: {doc_id}, Score: {score}")
    standardRanking  = getStandardRanking(query_id)
    k = 3
    ndcg_score = calculate_ndcg_score(sorted_ranking, standardRanking, k)
    # print("NDCG score:", ndcg_score)