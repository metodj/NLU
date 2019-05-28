from itertools import islice


outfile = open("BERT_pretraining_data.txt", 'w', encoding="utf-8")

with open ("train_stories.txt", encoding="utf-8") as infile:
    # Omit first line containing headers
    for line in islice(infile, 1, None):
        split_line = line.split('\t')
        # Omit story ID and story title
        for token in islice(split_line, 2, None):
            outfile.write(token + '\n')

outfile.close()
