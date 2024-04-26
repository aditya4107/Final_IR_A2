import os


def extract_words_from_tsv(filename):
    word_list = []

    try:
        # Open the file in read mode
        with open(filename, 'r', encoding='utf-8') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line by tabs to separate fields
                fields = line.strip().split('\t')
                # The first field contains the word
                word = fields[0]
                # Append the word to the list
                word_list.append(word)
    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")
        return None

    return word_list

current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
Allwords = extract_words_from_tsv(index_combined_file)

print(len(Allwords))

