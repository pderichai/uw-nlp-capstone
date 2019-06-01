#!/usr/bin/env python3

import re
from pytorch_pretrained_bert.tokenization import BertTokenizer


PUNCT_CHARS = {'.', '!', '?'}


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    with open('readcomp/dataprep/lamb/valid.txt.ner', 'r') as f, open('readcomp/dataprep/lamb/valid-bert.txt', 'w') as out:
        for line in f:
            groups = [extract_ner(g) for g in line.split()]
            tokens = [g[0] for g in groups]
            ner_tags = [g[1] for g in groups]
            ner_idx = 0
            sentence = []
            token_to_ner_idx = {}
            for idx in range(len(tokens)):
                token = tokens[idx]
                tag = ner_tags[idx]

                if tag != 'O':
                    if token not in token_to_ner_idx:
                        token_to_ner_idx[token] = ner_idx
                        ner_idx += 1
                else:
                    token_to_ner_idx[token] = -1

                word_pieces = tokenizer.tokenize(token)
                for wp in word_pieces:
                    sentence.append(wp + '/' + tag + '/' + str(token_to_ner_idx[token]))
                if idx == len(tokens) or ((token in PUNCT_CHARS and tokens[idx+1] != '\'\'') or (token == '\'\'' and tokens[idx-1] in PUNCT_CHARS)):
                    print(' '.join(sentence), file=out)
                    sentence = []
            print('', file=out)


def extract_ner(word):
    match = re.match('(.*)(\/([A-Z]*))(\|\|\|(.*))?', word.strip())
    return match.group(1), match.group(3), match.group(5) if len(match.groups()) >= 5 else None


if __name__ == '__main__':
    main()
