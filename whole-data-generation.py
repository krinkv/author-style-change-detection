import os
import json

DATA_PREFIX = 'problem-'
METADATA_PREFIX = 'truth-problem'
PARAGRAPH_SEPARATOR = '\n'
PARAGRAPH_SPLIT_REGEX = '\n'
RUN_OVER_FULL_DATA = True
PARTIAL_DATA_INDEX = 'ALL'
PARTIAL_DATA_VALIDATION_INDEX = 'ALL'
DATA_CATEGORIES = ['easy', 'medium', 'hard']
CATEGORY_KEY = '<category>'
DATA_PARTITION = '<size>'
TRAIN_DATA_PATH_TEMPLATE = './<category>/train'
VALIDATION_DATA_PATH_TEMPLATE = './<category>/validation'
IDENTIFIERS_PATH_TEMPLATE = './preprocessed/<category>-all-input-identifiers-single-file.txt'
TRAIN_DATA_OUTPUT_PATH_TEMPLATE = './preprocessed/<category>-<size>-train-data-single-file.txt'
VALIDATION_DATA_OUTPUT_PATH_TEMPLATE = './preprocessed/<category>-<size>-val-data-single-file.txt'
TRAIN_LABELS_OUTPUT_PATH_TEMPLATE = './preprocessed/<category>-<size>-train-labels-single-file.txt'
VALIDATION_LABELS_OUTPUT_PATH_TEMPLATE = './preprocessed/<category>-<size>-val-labels-single-file.txt'

def generate_pair_from_single_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    paragraphs = content.split(PARAGRAPH_SPLIT_REGEX)

    concatenated_paragraphs = []
    for i in range(len(paragraphs) - 1):
        concatenated_paragraphs.append(paragraphs[i] + PARAGRAPH_SEPARATOR + paragraphs[i + 1])

    return concatenated_paragraphs


def generate_pair_paragraphs(directory, partial_data_index):
    generated_input_pairs = []
    generated_input_identifiers = []
    count = 0
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        if (filename.startswith(DATA_PREFIX)):
            count += 1
            pairs_from_file = generate_pair_from_single_file(file_path)
            temp = 0
            for i in pairs_from_file:
                temp += 1
                generated_input_pairs.append(i)
                generated_input_identifiers.append(filename + '_pair_' + str(temp))

        if (not RUN_OVER_FULL_DATA) and count > partial_data_index:
            break

    return generated_input_pairs, generated_input_identifiers

def write_data_to_file(file_path, data):
  with open(file_path, 'w') as file:
      for string in data:
          file.write(string + '\n')

def get_pair_labels(directory, partial_data_index):
   pair_labels = []
   count = 0
   for filename in sorted(os.listdir(directory)):
      file_path = os.path.join(directory, filename)

      if (filename.startswith(METADATA_PREFIX)):
          count += 1
          parsed_label_file = parse_metadata_file(file_path)
          pair_labels.append(parsed_label_file['changes'])

      if (not RUN_OVER_FULL_DATA) and count > partial_data_index:
          break

   return pair_labels

def parse_metadata_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def flat_map(func, iterable):
    return [item for sublist in map(func, iterable) for item in sublist]

def write_labels_to_file(file_path, labels):
  with open(file_path, 'w') as file:
      for integer in labels:
          file.write(str(integer) + '\n')


def main():
    print(TRAIN_DATA_PATH_TEMPLATE.replace(CATEGORY_KEY, DATA_CATEGORIES[0]))

    for category in DATA_CATEGORIES:
        raw_training_data, input_identifiers = generate_pair_paragraphs(TRAIN_DATA_PATH_TEMPLATE.replace(CATEGORY_KEY, category), PARTIAL_DATA_INDEX)
        raw_validation_data, any = generate_pair_paragraphs(VALIDATION_DATA_PATH_TEMPLATE.replace(CATEGORY_KEY, category), PARTIAL_DATA_VALIDATION_INDEX)

        train_data_output_file_path = TRAIN_DATA_OUTPUT_PATH_TEMPLATE.replace(CATEGORY_KEY, category).replace(DATA_PARTITION, str(PARTIAL_DATA_INDEX))
        val_data_output_file_path = VALIDATION_DATA_OUTPUT_PATH_TEMPLATE.replace(CATEGORY_KEY, category).replace(DATA_PARTITION, str(PARTIAL_DATA_INDEX))
        identifiers_file_path = IDENTIFIERS_PATH_TEMPLATE.replace(CATEGORY_KEY, category)
        write_data_to_file(train_data_output_file_path, raw_training_data)
        write_data_to_file(val_data_output_file_path, raw_validation_data)
        write_data_to_file(identifiers_file_path, input_identifiers)

        raw_training_labels = flat_map(lambda x: x, get_pair_labels(TRAIN_DATA_PATH_TEMPLATE.replace(CATEGORY_KEY, category), PARTIAL_DATA_INDEX))
        raw_validation_labels = flat_map(lambda x: x, get_pair_labels(VALIDATION_DATA_PATH_TEMPLATE.replace(CATEGORY_KEY, category), PARTIAL_DATA_VALIDATION_INDEX))

        train_labels_output_file_path = TRAIN_LABELS_OUTPUT_PATH_TEMPLATE.replace(CATEGORY_KEY, category).replace(DATA_PARTITION, str(PARTIAL_DATA_INDEX))
        val_labels_output_file_path = VALIDATION_LABELS_OUTPUT_PATH_TEMPLATE.replace(CATEGORY_KEY, category).replace(DATA_PARTITION, str(PARTIAL_DATA_INDEX))
        write_labels_to_file(train_labels_output_file_path, raw_training_labels)
        write_labels_to_file(val_labels_output_file_path, raw_validation_labels)


if __name__ == "__main__":
    main()