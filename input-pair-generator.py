from transformers import DistilRobertaTokenizer

# Instantiate the tokenizer
tokenizer = DistilRobertaTokenizer.from_pretrained('distilroberta-base')

# Input text
input_text = "Hello, how are you doing today?"

# Tokenize input text
tokens = tokenizer(input_text, return_tensors='pt')

print("Input Text:", input_text)
print("Tokens:", tokens)
