import tokenization
import re
from scipy import spatial

numberbatch_vectors = []

def run():
    bert_dir = "..\\..\\..\\bert-master\\data\\data_pp\\"
    output_dir = "output\\data_pp\\"
    input_file = bert_dir + "sct.test.tsv"
    output_file = output_dir + "sct.test.results.tsv"

    set_type = "test"

    numberbatch_dir = "..\\..\\..\\numberbatch\\"
    numberbatch_file = numberbatch_dir + "numberbatch-en.txt"

    # This takes 4 GB of memory, but does not require a pandas or pytables installation.
    global numberbatch_vectors
    numberbatch_vectors = read_numberbatch(numberbatch_file)

    # Expected input format: id \t label \t text_a (s1..s4) \t text_b (e) 
    # TODO: Get input with non-concatenated sentences if possible
    examples = read_examples(input_file, set_type)

    results = [(guid, label, compute_distance(S, e)) for (guid, S, e, label) in examples]
    write_results(results, output_file, set_type)

# File handling
# -------------

def read_examples(input_file, set_type):
    with open(input_file) as file:
        lines = file.read().splitlines()

    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue

        line = line.split("\t")
        # TODO: Tokenize, delete stop words using NLTK/CoreNLP
        guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
        S = split_on_punctuation(tokenization.convert_to_unicode(line[2]))[:4]
        e = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
        examples.append((guid, S, e, label))
    return examples

def read_numberbatch(input_file):
    vectors = {}
    with open(input_file, encoding='latin-1') as file:
        for (i, line) in enumerate(file):
            if i == 0:
                continue

            line = line.split(" ")
            word = line[0]
            vector = [float(v) for v in line[1:]]
            vectors[word] = vector
    return vectors

def write_results(results, output_file, set_type):
    with open(output_file, "w") as file:

        for (guid, label, distance) in results:
            line = guid + "\t" + label + "\t" + ", ".join(str(d) for d in distance) + "\n"
            file.write(line)

# Tokenization
# ------------

def split_on_punctuation(sentences):
    return [s.strip() for s in re.split(r"[\.!?]", sentences) if s.strip() != ""]

def tokenize(sentence):
    # TODO: Tokenize, remove stopwords using NLTK/CoreNLP
    return sentence.strip().lower().split(" ");

def stem(word):
    # TODO: Stem word
    return word

# Vectors
# -------

def compute_distance(S, e):
    distance = []
    for s_j in S:
        distance_j = 0 
        num = 0
        for w in tokenize(e):
            max_d = max(cosine_similarity(w, u) for u in tokenize(s_j) if stem(w) != stem(u))
            num += 1

            distance_j += max_d 
        distance_j /= num
        distance.append(distance_j)
    return distance

def get_vector(word):
    if word in numberbatch_vectors:
        return numberbatch_vectors[word]

    return []

def cosine_similarity(word1, word2):
    vector1, vector2 = get_vector(word1), get_vector(word2)

    # Abort if words not in list
    if not vector1 or not vector2:
        return 0

    distance = spatial.distance.cosine(vector1, vector2)
    return 1 - distance

# Main
# ----

if __name__ == "__main__":
    run()