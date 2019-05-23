"""
Given a configuration, return the tokenized sentences to pre-contextualize.
"""
import argparse
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    all_documents = []
    current_document = []
    for line in tqdm(open(args.dataset_path)):
        # Strip newlines from lines
        line = line.rstrip("\n")
        # Skip sentence-separating lines.
        if line == "":
            continue

        # Check if the line is a docstart token
        # If so, append the current document to all documents,
        # reset it, and skip the token.
        token, pos, chunk, ner = line.split(" ")
        if token == "-DOCSTART-":
            if len(current_document) > 0:
                all_documents.append(current_document)
            current_document = []
            continue
        current_document.append(token)

    # Write the output to a file
    with open(args.output_path, "w") as output_file:
        for document in all_documents:
            output_file.write("{}\n\n".format(" ".join(document)))
    logger.info("Wrote output to {}".format(args.output_path))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=("Given a path to conll2003-formatted data, "
                     "return a text file with the documents in dataset."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help=("Path to CoNLL-X formatted CoNLL 2003 NER data."))
    parser.add_argument("--output-path", type=str, required=True,
                        help=("Path write the dataset raw text."))
    args = parser.parse_args()
    main()
