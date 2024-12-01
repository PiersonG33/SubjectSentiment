from itertools import chain
import nltk
# Run at least once with the following lines uncommented to download the necessary nltk data
# nltk.download('punkt_tab')

def levenshtein(s1, s2):
    # Calculate the Levenshtein distance between two strings
    # This is the minimum number of single-character edits (insertions, deletions, or substitutions)
    # required to change one string into the other
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def flatten_list(l):
    # Flatten a list of lists, or list of list of lists, etc
    flattened = []
    for item in l:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

# The article has to be split into sentences
# These could end with a period, exclamation mark, or question mark
# They could end by a newline, or be contained in quotes, too
def custom_split(text):
    # Split the text into sentences using the nltk library
    sentences = nltk.sent_tokenize(text)

    for i in range(len(sentences) - 1):
        # If the sentence ends with a quote, and the next sentence starts with a quote, combine them
        if sentences[i].endswith('"') and sentences[i + 1].startswith('"'):
            sentences[i] = sentences[i] + " " + sentences[i + 1]
            sentences[i + 1] = ""

    for i in range(len(sentences)):
        sentences[i] = sentences[i].split('\n')
        # Remove any empty strings
        sentences[i] = [line.strip() for line in sentences[i] if line.strip() != ""]
    sentences = flatten_list(sentences)
    
    # Debugging, ignore
    with open("output2.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    return sentences

# Takes in the article body text and a list of subjects
# Creates a dictionary, where the key is the subject
# and the value is a list of sentences that contain the subject
def divide_by_subject(article, subjects):
    sentences = custom_split(article)
    subdict = dict()
    for i, s in enumerate(sentences):
        for sub in subjects:
            if sub.lower() in s.lower():
                # Capture the sentence before and after the subject, if possible
                element = ""
                if i != 0:
                    element += sentences[i - 1]
                    if element[-1].isalnum():
                                element += ". "
                element += s
                if element[-1].isalnum():
                                element += ". "
                if i != len(sentences) - 1:
                    element += " " + sentences[i + 1]

                if sub in subdict:
                    subdict[sub].append(element)
                else:
                    subdict[sub] = [element]
            else:
                # Check by Levenshtein distance
                words = [word.strip() for word in s.split()]
                for word in words:
                    if levenshtein(sub.lower(), word.lower().strip()) <= 2:
                        element = ""
                        if i != 0:
                            element += sentences[i - 1]
                            if element[-1].isalnum():
                                element += ". "
                        element += s
                        if element[-1].isalnum():
                                element += ". "
                        if i != len(sentences) - 1:
                            element += " " + sentences[i + 1]

                        if sub in subdict:
                            subdict[sub].append(element)
                        else:
                            subdict[sub] = [element]
                        break
    return subdict