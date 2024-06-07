from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
import string
import contractions
import math
import textstat

stop_words = set(stopwords.words('english'))
punctuation = [',', '.', '!', '?', '"', '\'', '$', '%', ':', ';', '@']
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
            'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


def avg_occurrence_of_words(text):
    word_dict = {}
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if (token not in stop_words and token not in string.punctuation)]

    for token in tokens:
        if token in word_dict:
            word_dict[token] += 1
        else:
            word_dict[token] = 1

    return [sum(word_dict.values()) / len(word_dict)]


# recognize contracted word forms, such as 'they're' -> 'they are'
def number_of_contractions(text):
    num = 0
    tokens = custom_tokenize(text)

    for token in tokens:
        expanded_token = contractions.fix(token)
        if expanded_token != token:
            num += 1

    return [num]


# helper for number_of_contractions
def remove_leading_trailing_punctuation(word):
    while True:
        if word[0] in string.punctuation:
            word = word[1:]
        elif word[-1] in string.punctuation:
            word = word[:-1]
        else:
            break

    return word


def custom_tokenize(text):
    splitted = text.split(' ')

    for i in range(len(splitted)):
        splitted[i] = remove_leading_trailing_punctuation(splitted[i])

    return splitted


def count_stop_words_normalized(text):
    num_of_stop_words = 0
    words = nltk.word_tokenize(text)
    for word in words:
        if word in stop_words:
            num_of_stop_words += 1

    return [num_of_stop_words / len(words)]


def count_punctuation(text):
    feature_vector = [0 for _ in range(len(punctuation))]

    for i in range(len(text)):
        if text[i] in punctuation:
            feature_vector[punctuation.index(text[i])] += 1

    return feature_vector


def count_numbers(text):
    num_of_numbers = 0
    words = nltk.word_tokenize(text)
    for word in words:
        if any(char.isdigit() for char in word):
            num_of_numbers += 1

    return [num_of_numbers]


def pos_tag_statistics(text):
    feature_vector = [0 for _ in range(len(pos_tags))]
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)

    for tag in tagged_words:
        if tag[1] in pos_tags:
            feature_vector[pos_tags.index(tag[1])] += 1

    return feature_vector


def word_statistics(text):
    short = 0
    long = 0
    sum_chars = 0
    capitalized = 0
    all_caps = 0
    words = custom_tokenize(text)

    for word in words:
        len_w = len(word)
        sum_chars += len_w
        if len_w == 2 or len_w == 3:
            short += 1
        elif len_w > 7:
            long += 1

        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            capitalized += 1

        if word.isupper():
            all_caps += 1

    average_length = sum_chars / len(words)

    sentences = sent_tokenize(text)
    capitalized = max(0, capitalized - len(sentences))

    return [short, long, average_length, capitalized, all_caps]


def sentence_statistics(text):
    sentences = sent_tokenize(text)

    short = 0
    long = 0
    num_of_words = 0
    question_marks = 0
    exclamation_marks = 0
    periods = 0

    for sent in sentences:
        length = len(sent.split(' '))
        num_of_words += length
        if length < 5:
            short += 1
        elif length > 15:
            long += 1

        if sent.endswith('?'):
            question_marks += 1
        elif sent.endswith('!'):
            exclamation_marks += 1
        elif sent.endswith('...'):
            periods += 1

    average_length = num_of_words / len(sentences)

    return [short, long, average_length, question_marks, exclamation_marks, periods]


# helper in vocab_richness
def get_common_words():
    with open('./common-words.txt', 'r') as file:
        word_to_freq = {}
        lines = file.readlines()
        for line in lines:
            elems = line.split()
            word_to_freq[elems[0].lower()] = int(elems[-1])

        return word_to_freq


def vocab_richness(text):
    common_words = get_common_words()
    most_common_word_freq = common_words['the']
    text_words = custom_tokenize(text)

    sum_of_freq = 0
    unknown = 0

    for word in text_words:
        word = word.lower()
        if word in common_words:
            freq = math.log2(most_common_word_freq / common_words[word])
            sum_of_freq += freq
        else:
            unknown += 1

    average_freq = sum_of_freq / len(text_words)

    return [average_freq, unknown]


def readability_statistics(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    smog_index = textstat.smog_index(text)
    gunning_fog = textstat.gunning_fog(text)
    automated_readability_index = textstat.automated_readability_index(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    linsear_write_formula = textstat.linsear_write_formula(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)

    return [flesch_reading_ease, flesch_kincaid_grade, smog_index,
            gunning_fog, automated_readability_index, coleman_liau_index,
            linsear_write_formula, dale_chall_readability_score]


stylometry_functions = [avg_occurrence_of_words, number_of_contractions, count_stop_words_normalized,
                        count_punctuation, count_numbers, pos_tag_statistics, word_statistics,
                        sentence_statistics, vocab_richness, readability_statistics]


def get_stylometry_vector(text):
    feature_vector = []

    for func in stylometry_functions:
        feature_vector.extend(func(text))

    return feature_vector
