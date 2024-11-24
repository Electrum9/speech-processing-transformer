import random
import re

# Function to corrupt the word phonetically
def corrupt_word(word, corruption_prob=0.3):
    """
    Corrupts a word by probabilistically substituting its characters with phonetically similar ones.
    """
    phonetic_map = {
        'b': 'p', 'c': 'k', 'd': 't', 'e': 'i', 'f': 'v', 'g': 'j',
        'h': 'x', 'i': 'y', 'j': 'g', 'k': 'c', 'l': 'r', 'm': 'n',
        'n': 'm', 'o': 'u', 'p': 'b', 'q': 'k', 'r': 'l', 's': 'z',
        't': 'd', 'u': 'o', 'v': 'f', 'w': 'v', 'x': 'h', 'y': 'i', 'z': 's'
    }
    corrupted_word = []

    for char in word:
        # Probabilistically corrupt each character
        if char.lower() in phonetic_map and random.random() < corruption_prob:
            substitute = phonetic_map[char.lower()]
            # Preserve case of original character
            substitute = substitute.upper() if char.isupper() else substitute
            corrupted_word.append(substitute)
        else:
            corrupted_word.append(char)  # Leave unchanged

    return ''.join(corrupted_word)

# Function to corrupt the entire text with phonetic similarity
def corrupt_text_with_phonetics(text, corruption_prob=0.3):
    """
    Applies phonetic corruption to all words in the input text.
    """
    words = text.split()
    corrupted_words = [corrupt_word(word, corruption_prob) for word in words]
    return ' '.join(corrupted_words)

# Function to swap language markers '[ITA]' and '[ENG]' with some probability
def corrupt_language_marker(text, prob=0.5):
    """
    Swaps '[ITA]' with '[ENG]' probabilistically.
    """
    if '[ITA]' in text or '[ENG]' in text:
        if random.random() < prob:
            text = text.replace('[ITA]', '[TEMP]')
            text = text.replace('[ENG]', '[ITA]')
            text = text.replace('[TEMP]', '[ENG]')
    return text

# Function to preserve identifier parts (e.g., "test_01081") and apply corruption to the rest of the text
def corrupt_line_with_preserved_identifier(line, corruption_prob=0.3, lang_swap_prob=0.5):
    """
    Corrupts text while preserving identifier parts (e.g., 'test_00000 [ITA]').
    """
    identifier_pattern = r"test_\d+"  # Matches patterns like 'test_00000'

    # Split the line into parts while preserving identifiers
    parts = re.split(f"({identifier_pattern})", line)

    # Apply corruption to non-identifier parts
    for i in range(len(parts)):
        if not re.match(identifier_pattern, parts[i]):  # Skip the identifiers
            parts[i] = corrupt_language_marker(parts[i], prob=lang_swap_prob)
            parts[i] = corrupt_text_with_phonetics(parts[i], corruption_prob)

    return ''.join(parts)

# Function to process a file line by line
def process_file(input_file_path, output_file_path, corruption_prob=0.3, lang_swap_prob=0.5):
    """
    Processes a file line by line, applying phonetic and marker corruption while preserving identifiers.
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Process each line with the corruption function
            corrupted_line = corrupt_line_with_preserved_identifier(line.strip(), corruption_prob, lang_swap_prob)
            outfile.write(corrupted_line + '\n')

# Example of how to run the script
if __name__ == "__main__":
    # User input for file paths and probabilities
    input_file = input("Enter the path to the input file: ").strip()
    output_file = input("Enter the path to the output file: ").strip()
    phonetic_probability = float(input("Enter phonetic corruption probability (default 0.3): ") or 0.3)
    language_swap_probability = float(input("Enter language marker swap probability (default 0.5): ") or 0.5)

    # Process the file
    process_file(input_file, output_file, phonetic_probability, language_swap_probability)
    print(f"Corrupted text saved to {output_file}")
