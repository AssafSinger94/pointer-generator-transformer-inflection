import argparse
import os

import data

parser = argparse.ArgumentParser(description='Reads conll dev file, and covers the target (test file format)')
parser.add_argument('--data-dir', type=str, default='train',
                    help="Folder of the dataset file")
parser.add_argument('--lang', type=str, default='train',
                    help="language of the dataset file")
args = parser.parse_args()


def cover_file(data_dir, language):
    dev_file_path = os.path.join(data_dir, f"{language}.dev")
    covered_dev_file_path = os.path.join(data_dir, f"{language}.covered_dev")
    covered_dev_file = open(covered_dev_file_path, "w", encoding='utf-8')  # "ISO-8859-1")
    dev_morph_list = data.read_morph_file(dev_file_path)
    for lemma, target, feature in dev_morph_list:
        covered_dev_file.write(f"{lemma}\t{feature}\n")
    covered_dev_file.close()


if __name__ == '__main__':
    # Create vocab files
    cover_file(args.data_dir, args.lang)