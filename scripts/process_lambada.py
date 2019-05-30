#!/usr/bin/env python3

import re
from pytorch_pretrained_bert.tokenization import BertTokenizer


PUNCT_CHARS = ['.', '!', '?']


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    with open('readcomp/dataprep/lamb/valid.txt.ner', 'r') as f, open('readcomp/dataprep/lamb/valid-bert.txt', 'w') as out:
        for line in f:
            groups = [extract_ner(g) for g in line.split()]
            tokens = [g[0] for g in groups]
            ner_tags = [g[1] for g in groups]
            idx = 0
            sentence = []
            while idx < len(tokens):
                token = tokens[idx]
                tag = ner_tags[idx]
                word_pieces = tokenizer.tokenize(token)
                for wp in word_pieces:
                    sentence.append(wp + '/' + tag)
                if idx == len(tokens) or ((token in PUNCT_CHARS and tokens[idx+1] != '\'\'') or (token == '\'\'' and tokens[idx-1] in PUNCT_CHARS)):
                    #print(token, tokens[idx+1])
                    print(' '.join(sentence), file=out)
                    sentence = []
                idx += 1
            print('', file=out)


def extract_ner(word):
    match = re.match('(.*)(\/([A-Z]*))(\|\|\|(.*))?', word.strip())
    return match.group(1), match.group(3), match.group(5) if len(match.groups()) >= 5 else None


if __name__ == '__main__':
    main()
