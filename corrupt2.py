import random
import re

# Function to corrupt the word phonetically
def corrupt_word(word, corruption_prob=0.3):
    if random.random() > corruption_prob:
        return word  # If the random value is higher than corruption_prob, leave the word unchanged

    phonetic_map = {
        'b': 'p', 'c': 'k', 'd': 't', 'e': 'i', 'f': 'v', 'g': 'j',
        'h': 'x', 'i': 'y', 'j': 'g', 'k': 'c', 'l': 'r', 'm': 'n',
        'n': 'm', 'o': 'u', 'p': 'b', 'q': 'k', 'r': 'l', 's': 'z',
        't': 'd', 'u': 'o', 'v': 'f', 'w': 'v', 'x': 'h', 'y': 'i', 'z': 's'
    }

    corrupted_word = ''.join([phonetic_map.get(char, char) for char in word])

    return corrupted_word

# Function to corrupt the entire text with phonetic similarity
def corrupt_text_with_phonetics(text, corruption_prob=0.3):
    words = text.split()
    corrupted_words = [corrupt_word(word, corruption_prob) for word in words]
    return ' '.join(corrupted_words)

# Function to swap language markers '[ITA]' and '[ENG]' with some probability
def corrupt_language_marker(text, prob=0.5):
    text = text.replace('[ITA]', '[TEMP]')
    text = text.replace('[ENG]', '[ITA]')
    text = text.replace('[TEMP]', '[ENG]')

    if random.random() < prob:
        text = text.replace('[ITA]', '[TEMP]')
        text = text.replace('[ENG]', '[ITA]')
        text = text.replace('[TEMP]', '[ENG]')

    return text

# Function to preserve identifier parts (e.g., "test_01081") and apply corruption to the rest of the text
def corrupt_line_with_preserved_identifier(line, corruption_prob=0.3, lang_swap_prob=0.5):
    identifier_pattern = r"test_\d+"

    # Split the line into words, but preserve the identifiers
    parts = re.split(f"({identifier_pattern})", line)

    # Apply corruption to non-identifier parts
    for i in range(len(parts)):
        if not re.match(identifier_pattern, parts[i]):  # Skip the identifiers
            parts[i] = corrupt_language_marker(parts[i], prob=lang_swap_prob)
            parts[i] = corrupt_text_with_phonetics(parts[i], corruption_prob)

    return ''.join(parts)

# Function to process a file line by line
def process_file(input_file_path, output_file_path, corruption_prob=0.3, lang_swap_prob=0.5):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Process each line with the corruption function
            corrupted_line = corrupt_line_with_preserved_identifier(line.strip(), corruption_prob, lang_swap_prob)
            outfile.write(corrupted_line + '\n')
