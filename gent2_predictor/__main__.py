import argparse

import torch

from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.predictor.ffn import FFN
from gent2_predictor.predictor.generic_transformer import Transformer
from gent2_predictor.predictor.trainer import Trainer
from gent2_predictor.settings import MODEL_PATH


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

    parser.add_argument(
        '-tt', '--transformer_train', action='store_true',
        help="Train a transformer deep neural network")

    args = parser.parse_args()

    if args.parse:
        DataParser().parse_structure()
        DataParser().pickle_data()

    elif args.ffn_train:
        model = FFN()
        trainer = Trainer(model)
        trainer.start_loop()

    elif args.predict_on_ffn:
        model = FFN()
        model.load_state_dict(torch.load(MODEL_PATH))
        trainer = Trainer(model)
        scores = trainer.predict()
        print(scores)

    elif args.transformer_train:
        model = Transformer(
            d_model=10000, nhead=1, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=100)
        trainer = Trainer(model)
        trainer.start_loop()


if __name__ == "__main__":
    main()
