tags = ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-ORG', 'B-LOC', 'B-MISC', 'O']

import json
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def parse_expectations(filename):
    result = []
    f = open(filename)
    for line in f.readlines():
        if '-DOCSTART-' in line:
            continue
        if line == '\n':
            continue
        _, _, _, tag = line.split()
        result.append(tag)
    f.close()
    return result

def parse_predictions(filename):
    result = []
    f = open(filename)
    for line in f.readlines():
        line_obj = json.loads(line)
        result += line_obj["tags"]
    f.close()
    return result

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    if len(sys.argv) != 3:
        print("Usage: python get_confusion_matrix.py <gold_standard> <predicted_file>")
        return
    predictions = parse_predictions(sys.argv[2])
    expectations = parse_expectations(sys.argv[1])
    cnf_matrix = confusion_matrix(expectations, predictions, labels=tags);
    plot_confusion_matrix(cnf_matrix, tags, True)
    plt.show()

if __name__ == '__main__':
    main()
