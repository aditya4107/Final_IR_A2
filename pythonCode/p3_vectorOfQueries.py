from utilityFunctions import retrieve_document_vector_values as getDocVector

vector = getDocVector("MED-1")

def get_word_value(vector, word):
    return vector.get(word, 0.0)

word = "aa"
value = get_word_value(vector, word)
print(f"Value of '{word}' in the vector: {value}")
