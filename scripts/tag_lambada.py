#!/usr/bin/env python3

import mmap
import argparse

from tqdm import tqdm
import torch
import allennlp
from allennlp.pretrained import named_entity_recognition_with_elmo_peters_2018


PUNCT_CHARS = {'.', '!', '?'}


def main():
    parser = argparse.ArgumentParser(description=('Tag the LAMBADA data'))
    parser.add_argument('dataset_path')
    parser.add_argument('output_path')

    args = parser.parse_args()

    ner_model = named_entity_recognition_with_elmo_peters_2018()
    ner_model._model = ner_model._model.to(device=torch.device('cuda'))
    ner_model._tokenizer = allennlp.data.tokenizers.word_splitter.JustSpacesWordSplitter()

    with open(args.dataset_path, 'r') as lambada, open(args.output_path, 'w') as out:
        for passage in tqdm(lambada, total=get_num_lines(args.dataset_path)):
            tokens = passage.split()
            sentences = []
            prev_idx = -1
            for idx in range(len(tokens)):
                token = tokens[idx]
                # sentence segmentation
                if idx == len(tokens)-1 or ((token in PUNCT_CHARS and tokens[idx+1] != '\'\'') or (token == '\'\'' and tokens[idx-1] in PUNCT_CHARS)):
                    sentences.append(' '.join(tokens[prev_idx+1:idx+1]))
                    prev_idx = idx

            for sentence in sentences:
                results = ner_model.predict(sentence=sentence)
                list_to_print = []
                for word, tag in zip(results['words'], results['tags']):
                    list_to_print.append(word + '/' + tag)
                print(' '.join(list_to_print), file=out)
            print('', file=out)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


if __name__ == '__main__':
    main()
