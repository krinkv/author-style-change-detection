import os
import re
import json
import matplotlib.pyplot as plt

DATA_PREFIX = 'problem-'
METADATA_PREFIX = 'truth-problem'

def count_file_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    paragraphs = content.split('\n')
    
    return len(paragraphs)  


def get_files_paragraphs_count(directory):
    files = []
    files_paragraphs_count = []

    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        if (filename.startswith(DATA_PREFIX)):
            num_paragraphs = count_file_paragraphs(file_path)
            files.append(filename)
            files_paragraphs_count.append(num_paragraphs)

    return (files, files_paragraphs_count)  

def get_files_paragraphs_count_v2(directory):
    files = []
    files_paragraphs_count = []

    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        if (filename.startswith(METADATA_PREFIX)):
            metadata = parse_metadata_file(file_path)
            num_paragraphs = len(metadata['changes']) + 1
            files.append(filename)
            files_paragraphs_count.append(num_paragraphs)

    return (files, files_paragraphs_count)          

def get_all_paragraphs_count(files_paragraphs_count):
    return sum(files_paragraphs_count)            

def draw_plot(files, files_paragraphs_count):
    plt.figure(figsize=(10, 5))
    plt.bar(files, files_paragraphs_count, color='skyblue')
    plt.xlabel('Files')
    plt.ylabel('Number of Paragraphs')
    plt.title('Paragraph Count per File')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def parse_metadata_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_all_author_changes(directory):
    author_changes = 0
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        if (filename.startswith(METADATA_PREFIX)):
            metadata = parse_metadata_file(file_path)
            author_changes += sum(1 for element in metadata['changes'] if element == 1)

    return author_changes        

def provide_noisy_files(files, files_paragraphs_count_correct, files_paragraphs_count_incorrect):
    for i in range(len(files_paragraphs_count_correct)):
        if (files_paragraphs_count_correct[i] != files_paragraphs_count_incorrect[i]):
            print(files[i])

def main():
    (metadata_files, files_paragraphs_count_correct) = get_files_paragraphs_count_v2('./data/pan24/easy/train')
    (files, files_paragraphs_count_incorrect) = get_files_paragraphs_count('./data/pan24/easy/train')
    
    provide_noisy_files(files, files_paragraphs_count_correct, files_paragraphs_count_incorrect)
    # (files, files_paragraphs_count) = get_files_paragraphs_count('./data/pan24/medium/train')
    # (files, files_paragraphs_count) = get_files_paragraphs_count('./data/pan24/hard/train')

    # print(get_all_paragraphs_count(files_paragraphs_count))

    # draw_plot(files, files_paragraphs_count)

    # print(count_all_author_changes('./data/pan24/easy/train'))

if __name__ == "__main__":
    main()




    ## Group all paragraphs in single file by pairs but group only these which we know if there is an author change or not
    ## Get all pairs and provide statistic about how many of them have more than 512 tokens
    ## https://chatgpt.com/share/631d7298-e836-4948-86fa-8a5196b495c1
    ## truncating by generating more pairs from single pair to guarantee that we use all the data


    # Strategies for Handling Important Data

    # Chunking: Split the text into overlapping or non-overlapping chunks of 512 tokens or less.
    # Summarization: Use a summarization model to condense the text before feeding it to BERT.
    # Sliding Window: Use a sliding window approach to create overlapping segments.
    # Hierarchical Models: Use hierarchical attention or multi-stage models to handle long texts.