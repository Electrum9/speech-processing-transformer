import random
import re

def corrupt_phonetically(word):
    """
    Generate a phonetic corruption of a given word by making random
    substitutions, insertions, or deletions to mimic phonetic resemblance.

    Args:
        word (str): The original word to corrupt.

    Returns:
        str: A corrupted version of the word.
    """
    vowels = "AEIOUY"
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    corruption_type = random.choice(["substitution", "insertion", "deletion"])

    if corruption_type == "substitution":
        # Randomly replace one character with another similar character
        idx = random.randint(0, len(word) - 1)
        if word[idx].upper() in vowels:
            replacement = random.choice(vowels.replace(word[idx].upper(), ""))
        else:
            replacement = random.choice(consonants.replace(word[idx].upper(), ""))
        corrupted = word[:idx] + replacement + word[idx + 1:]

    elif corruption_type == "insertion":
        # Insert a random character at a random position
        idx = random.randint(0, len(word))
        if random.random() > 0.5:
            char = random.choice(vowels)
        else:
            char = random.choice(consonants)
        corrupted = word[:idx] + char + word[idx:]

    elif corruption_type == "deletion":
        # Remove a random character from the word
        if len(word) > 1:
            idx = random.randint(0, len(word) - 1)
            corrupted = word[:idx] + word[idx + 1:]
        else:
            corrupted = word  # Avoid deleting the only character

    return corrupted

def replace_with_corruptions(text, corruption_prob=0.8):
    """
    Replace words in the given text with their corrupted counterparts.

    Args:
        text (str): Input text to corrupt.
        corruption_prob (float): Probability of corrupting a word (0 to 1).

    Returns:
        str: Text with corrupted words.
    """
    words = text.split()
    corrupted_text = [
        corrupt_phonetically(word.upper()) if word.isalpha() and random.random() < corruption_prob else word.upper()
        for word in words
    ]
    return " ".join(corrupted_text)

def process_file(input_file, output_file, corruption_prob=0.8):
    """
    Processes the file, applies corruption to each line (excluding non-alphabetic parts),
    and writes the output to another file.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        corruption_prob (float): Probability of corrupting a word (0 to 1).
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Process each line, leaving identifiers (test_...) intact
            parts = re.split(r'(\btest_\d+\s)', line)  # Split on identifiers like test_01081
            processed_line = []
            for part in parts:
                if part.startswith("test_"):
                    # Keep test identifiers as is
                    processed_line.append(part)
                else:
                    # Corrupt the words in the part of the line that is not an identifier
                    processed_line.append(replace_with_corruptions(part.strip(), corruption_prob))
            outfile.write("".join(processed_line) + "\n")
