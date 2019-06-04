#!/usr/bin/env python3

import re
from pytorch_pretrained_bert.tokenization import BertTokenizer
import argparse

PUNCT_CHARS = {'.', '!', '?'}


def main():
    parser = argparse.ArgumentParser(description=('Process the tagged LAMBADA data'))
    parser.add_argument('dataset_path')
    parser.add_argument('output_path')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    with open(args.dataset_path, 'r') as f, open(args.output_path, 'w') as out:
        for line in f:
            groups = [extract_ner(g) for g in line.split()]
            tokens = [g[0] for g in groups]
            ner_tags = [g[1] for g in groups]

            # sentence segmentation
            sentences = []
            per_sentence_ner_tags = []
            sentence = []
            sentence_ner_tags = []
            for idx in range(len(tokens)):
                token = tokens[idx]
                sentence.append(token)
                sentence_ner_tags.append(ner_tags[idx])
                if idx == len(tokens)-1 or ((token in PUNCT_CHARS and tokens[idx+1] != '\'\'') or (token == '\'\'' and tokens[idx-1] in PUNCT_CHARS)):
                    sentences.append(sentence)
                    per_sentence_ner_tags.append(sentence_ner_tags)
                    sentence = []
                    sentence_ner_tags = []

            # index the entities and write to output file
            ner_idxs = []
            for tokens, ner_tags in zip(sentences, per_sentence_ner_tags):
                sentence = []
                ner_idx = 1
                token_to_ner_idx = {}
                for idx in range(len(tokens)):
                    token = tokens[idx]
                    tag = ner_tags[idx]

                    word_pieces = tokenizer.tokenize(token)
                    if tag != 'O':
                        if token not in token_to_ner_idx:
                            token_to_ner_idx[token] = ner_idx
                            ner_idx += 1
                    else:
                        token_to_ner_idx[token] = 0
                    for wp in word_pieces:
                        sentence.append(wp + '|||' + tag + '|||' + str(token_to_ner_idx[token]))

                print(' '.join(sentence), file=out)
            print('', file=out)


def extract_ner(word):
    match = re.match('(.*)(\/([A-Z]*))(\|\|\|(.*))?', word.strip())
    return match.group(1), match.group(3), match.group(5) if len(match.groups()) >= 5 else None


if __name__ == '__main__':
    main()
