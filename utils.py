from re import sub

def split_to_dict(text: str, arrOfNames: list = ['topic', 'keyword']) -> dict:
    """
    Split a string into two words and store them in a dictionary with keys 'first' and 'second'.

    Args:
        string (str): The input string to split.
        arrOfNames (list): A list of the names of the words in the string.
    Returns:
        dict: A dictionary with keys 'first' and 'second' mapping to the two words.

    Raises:
        ValueError: If the input string does not contain exactly two words.
    """
    # Split the string into words based on whitespace
    global new_dict
    words = text.split()

    # Check if there are exactly two words
    print(words)

    # Store the words in a dictionary with named keys
    new_dict = dict()
    for index, word in enumerate(words):
        new_dict[arrOfNames[index]] =  sub('[^a-zA-Z]', '', word)
    return new_dict