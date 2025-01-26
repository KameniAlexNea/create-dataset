import os
import random
import re
import string
from glob import glob

punctuation = "?!."


def merge_sentences(sentence: str):
    sentences = sentence.strip().split("\n")
    result = sentences[0].strip()
    for i, j in zip(sentences, sentences[1:]):
        i, j = i.strip(), j.strip()  # Strip leading and trailing whitespace

        # If the line ends with a hyphen, merge without space to continue the word
        if i.endswith("-"):
            result += j
        # If the line ends with punctuation, start a new line
        elif i and i[-1] in punctuation:
            result += "\n" + j
        # If both lines start with uppercase, consider it a new paragraph/title
        elif j and i.isupper() and j[0].isupper():
            result += "\n" + j
        # Otherwise, add a space and continue the sentence
        else:
            result += " " + j if j else ""
    return result.strip()


def is_potential_title(line: str) -> bool:
    line = line.strip()  # Remove leading and trailing whitespace

    # Check if the line starts with a number followed by a period (e.g., "19.")
    if re.match(r"^\d+\.", line):
        return True

    # Other heuristics to identify if it's likely a title
    if len(line.split()) <= 10 and not line.endswith(
        tuple(string.punctuation)
    ):  # Short and no ending punctuation
        if line[0].isupper():  # Starts with an uppercase letter
            words = line.split()
            # Optional: Check for capitalized words in the title
            capitalized_words = sum(1 for word in words if word[0].isupper())
            # Allow a few lowercase words if there are capitalized ones
            if capitalized_words >= len(words) / 2:
                return True

    return False


def is_valid_file(raw: str):
    lines = [i for i in raw.split("\n") if i.strip()]
    count = [i for i in lines if ".............." in i]  # potential title
    count2 = [i for i in lines if "—." in i]  # potential citation
    count3 = [1 for i in lines if is_potential_title(i)]  # potential citation part
    return (
        (50 > len(lines) > 10)
        and len(count) < 5
        and len(count2) < 5
        and len(count3) < 5
    )


def get_dataset(folder: str):
    files = sorted(glob(os.path.join(folder, "*.txt")))

    data = [open(file).read() for file in files]

    data = [merge_sentences(i) for i in data if is_valid_file(i)]
    return data


def select_pages(pages, n, m):
    if not pages:
        return []
    pages = sorted(pages)

    results = []
    selected_pages = set()

    sample_pages = random.sample(pages, min(n * 3, len(pages)))
    for start_index in sample_pages:
        consecutive_pages = pages[start_index : start_index + m]

        # Check for overlap
        if all(page not in selected_pages for page in consecutive_pages):
            results.append(consecutive_pages)
            selected_pages.union(consecutive_pages)

        # Exit if we have enough results
        if len(results) >= n:
            break
    return results


def choose_context_pages(folder: str, n_questions: int, n_range: int):
    files = sorted(glob(os.path.join(folder, "*.txt")))
    pages = list(range(len(files)))
    selected_pages = select_pages(pages, n_questions, n_range)
    return [[files[j] for j in i] for i in selected_pages]
