import argparse

from gent2_predictor.data_parser.data_parser import DataParser


def main():
    parser = argparse.ArgumentParser(
        description="Deep learning package for Cancer type prediction"
                    "from microarray gene expression data")

    parser.add_argument(
        '-p', '--parse', action='store_true',
        help='parse the data and pickle them to file')

    parser.add_argument(
        '-tf', '--ffn_train', action='store_true',
        help="Train a fully connected deep neural network")

    parser.add_argument(
        '-pf', '--predict_on_ffn', action='store_true',
        help='Make predictions on the previously trained model')

    args = parser.parse_args()

    if args.parse:
        DataParser().parse_structure()
        DataParser().pickle_data()

    elif args.tf:
        raise NotImplementedError

    elif args.pf:
        raise NotImplementedError


if __name__ == "__main__":
    main()
