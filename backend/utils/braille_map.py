# utils/braille_map.py

# Braille dot patterns mapped to characters
# Each tuple represents which dots are raised (1-6)

BRAILLE_MAP = {
    # --- Letters ---
    (1,): 'a',
    (1, 2): 'b',
    (1, 4): 'c',
    (1, 4, 5): 'd',
    (1, 5): 'e',
    (1, 2, 4): 'f',
    (1, 2, 4, 5): 'g',
    (1, 2, 5): 'h',
    (2, 4): 'i',
    (2, 4, 5): 'j',
    (1, 3): 'k',
    (1, 2, 3): 'l',
    (1, 3, 4): 'm',
    (1, 3, 4, 5): 'n',
    (1, 3, 5): 'o',
    (1, 2, 3, 4): 'p',
    (1, 2, 3, 4, 5): 'q',
    (1, 2, 3, 5): 'r',
    (2, 3, 4): 's',
    (2, 3, 4, 5): 't',
    (1, 3, 6): 'u',
    (1, 2, 3, 6): 'v',
    (2, 4, 5, 6): 'w',
    (1, 3, 4, 6): 'x',
    (1, 3, 4, 5, 6): 'y',
    (1, 3, 5, 6): 'z',

    # --- Numbers (preceded by number indicator (3,4,5,6)) ---
    (2,): '1',
    (2, 3): '2',
    (2, 5): '3',
    (2, 5, 6): '4',
    (2, 6): '5',
    (2, 3, 5): '6',
    (2, 3, 5, 6): '7',
    (2, 3, 6): '8',
    (3, 5): '9',
    (3, 5, 6): '0',

    # --- Punctuation ---
    (2, 6): ',',
    (2, 5, 6): ';',
    (2, 5): ':',
    (2, 5, 6): '!',
    (2, 3, 5, 6): '?',
    (2, 3, 6): '-',
    (3, 4, 5, 6): '#',  # number indicator
    (6,): '',           # capital indicator (next letter is capital)

    # --- Grade 2 Contractions (common ones) ---
    (1, 2): 'but',
    (1, 6): 'can',
    (1, 4, 5, 6): 'do',
    (1, 5, 6): 'every',
    (1, 2, 4, 6): 'from',
    (1, 2, 4, 5, 6): 'go',
    (1, 2, 5, 6): 'have',
    (2, 4, 6): 'just',
    (1, 3, 6): 'knowledge',
    (1, 2, 3, 6): 'like',
    (1, 3, 4, 6): 'more',
    (1, 3, 4, 5, 6): 'not',
    (1, 3, 5, 6): 'people',
    (1, 2, 3, 4, 6): 'quite',
    (1, 2, 3, 5, 6): 'rather',
    (2, 3, 4, 6): 'so',
    (2, 3, 4, 5, 6): 'that',
    (1, 3, 4, 5, 6): 'us',
    (1, 2, 3, 6): 'very',
    (2, 4, 5, 6): 'will',
    (1, 3, 4, 6): 'it',
    (1, 3, 5, 6): 'you',
    (1, 3, 4, 5, 6): 'as',
}

# Special indicators
CAPITAL_INDICATOR = (6,)
NUMBER_INDICATOR = (3, 4, 5, 6)
SPACE = ()  # empty cell = space

def decode_pattern(pattern, next_is_capital=False):
    """
    Takes a dot pattern tuple and returns the English character.
    Handles capital indicator automatically.
    """
    pattern = tuple(sorted(pattern))

    if pattern == CAPITAL_INDICATOR:
        return None  # signals next char should be capital

    if pattern == SPACE:
        return ' '

    char = BRAILLE_MAP.get(pattern, '?')  # '?' if pattern not found

    if next_is_capital and char.isalpha():
        return char.upper()

    return char


def decode_sequence(patterns):
    """
    Takes a list of dot pattern tuples and returns a decoded string.
    Handles capital indicators in sequence.
    """
    result = []
    capitalize_next = False

    for pattern in patterns:
        char = decode_pattern(pattern, capitalize_next)

        if char is None:
            capitalize_next = True
            continue

        capitalize_next = False
        result.append(char)

    return ''.join(result)