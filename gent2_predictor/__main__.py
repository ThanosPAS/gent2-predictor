import argparse

from gent2_predictor.data_parser.data_parser import DataParser


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p', '--parse', action='store_true',
        help='parse the data and pickle them to file')

    args = parser.parse_args()

    if args.parse:
        data_parser = DataParser()
        data = data_parser.parse_cancer_files()
        data_parser.pickle_data(data)


if __name__ == "__main__":
    main()
