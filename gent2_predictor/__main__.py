import argparse
import argparse
import errno
import os

import torch
from gent2_predictor.predictor.ffn import FFNTrainer
from gent2_predictor.settings import DEVICE, MODEL_PATH_DIR, MODEL_PATH
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
        trainer = FFNTrainer()
        trainer.train_ffn()

        if not os.path.exists(MODEL_PATH_DIR):
            try:
                os.makedirs(MODEL_PATH_DIR)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        torch.save(trainer.model.state_dict(), MODEL_PATH)

    elif args.pf:
        pretrained_model = torch.load(MODEL_PATH)
        trainer = FFNTrainer(pretrained_model)
        scores = trainer.predict_ffn()
        print(scores)


if __name__ == "__main__":
    main()
