from collections import defaultdict
from math import log
import math
import os
from utilityFunctions import retrieve_document_vector_values as getDocVector

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



# loading the list of docids
docid_list = []

def retrieve_docids_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)

final_data_file = os.path.join(os.getcwd(),'pythonCode', 'concatData', 'finalData.txt')

if os.path.exists(final_data_file):
    retrieve_docids_from_file(final_data_file)
else:
    print(f"File not found: {final_data_file}")


# retrieve relevence data given a queryid
def load_relevance_data(file_path):
    relevance_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = []
            relevance_data[query_id].append((doc_id, relevance_score))
    return relevance_data

relevance_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'relevance')
file_path = os.path.join(relevance_folder, 'merged.qrel')
relevance_data = load_relevance_data(file_path)

def get_documents_with_scores(query_id):
    if query_id in relevance_data:
        return relevance_data[query_id]
    else:
        return []

def extract_words_from_text(text):
    words = text.split()
    return words

# sum of vectors
def sum_document_vectors(relevant_docs):
    summed_vector = defaultdict(float)
    for doc_id in relevant_docs:
        vector = getDocVector(doc_id)
        for word, value in vector.items():  # Iterate over items, not just keys
            # Ensure value is converted to float before adding
            summed_vector[word] += float(value)
    return sorted(summed_vector.items())

def compute_difference(relevant_vector, non_relevant_vector, beta, gamma):
    difference_vector = defaultdict(float)
    
    # Iterate over relevant vector
    for word, value in relevant_vector:
        difference_vector[word] += beta * value
    
    # Subtract non-relevant vector
    for word, value in non_relevant_vector:
        difference_vector[word] -= gamma * value
    
    return sorted(difference_vector.items())

def add_alpha_to_vector(vector, query_words, alpha):
    updated_vector = {}  # Create a new dictionary to store the updated values
    for word in query_words:
        if word in vector:
            updated_vector[word] = float(vector[word]) + alpha
        else:
            updated_vector[word] = alpha  # If word doesn't exist, add it with alpha as value
    
    # Make any negative values zero in the updated vector
    for word, value in updated_vector.items():
        if value < 0:
            updated_vector[word] = 0
    
    return updated_vector


def non_zero_words(vector):
    non_zero_dict = {}
    for word, value in vector.items():
        if value != 0:
            non_zero_dict[word] = value
    return non_zero_dict

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

# Load the index_combined and index_map
current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
index_combined = load_index_combined(index_combined_file)
index_map = build_index_map(index_combined_file)

# searches whether a word exists in a doc
def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0

# get the doc value for a vector
def getdocValue(word):
    if choice_doc == 1:
        return 1.0
    else:
        df =  index_combined[word][0]
        return log(8808 / df)
    
#  get the query value for a vector
def getqueryValue(word):
    if choice_query == '1':
        return 1.0
    else: 
        df = index_combined[word][0]
        return log(8808 / df)

# does normalisation
def cosine_normalization_term(vector):
    squared_sum = sum(x ** 2 for x in vector)
    normalization_term = math.sqrt(squared_sum)
    return normalization_term

# main task starts here:
for query in queryList:
    query_id, query_text = query
    queryWords = extract_words_from_text(query_text)
    documents = get_documents_with_scores(query_id)
    relevant_docs = []
    non_relevant_docs = []
    for doc_id, score in documents:
        if score <= 1:
            non_relevant_docs.append(doc_id)
        else:
            relevant_docs.append(doc_id)
    lenRel = len(relevant_docs)
    lenNonRel = len(non_relevant_docs)
    relevant_vectors = sum_document_vectors(relevant_docs)
    nonrelevant_vectors = sum_document_vectors(non_relevant_docs)
    alpha = 1
    beta = 0.75
    gamma = 0.25
    if(lenRel!=0):
        beta = beta/lenRel
    if(lenNonRel!=0):
        gamma = gamma/lenNonRel
    difference = compute_difference(relevant_vectors, nonrelevant_vectors, beta, gamma)
    updatedQuery = add_alpha_to_vector(difference, queryWords, alpha)
    non_zero_words_dict = non_zero_words(updatedQuery)
    ranking = {}
    for docid in docid_list:
        similarity = 0
        doclist = []
        querylist = []
        for word, value in non_zero_words_dict.items():
            value = check_word_in_document(word, docid, index_map)
            if value != 0:
                docval = value*getdocValue(word)
                queryval = getqueryValue(word)
                doclist.append(docval)
                querylist.append(queryval)
                similarity += docval * queryval
        if choice_doc == 3:
            doc_norm = cosine_normalization_term(doclist)
            similarity = similarity / doc_norm
        if choice_query == 3:
            query_norm = cosine_normalization_term(querylist)
            similarity = similarity / query_norm 
        ranking[docid] = similarity
    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    print(query_id)
    print(query_text)
    for doc_id, score in sorted_ranking[:10]:
        print(f"Document ID: {doc_id}, Score: {score}")