import argparse

import torch
import torch.nn as nn

from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.predictor.ffn import FFN, Baseline_FFN, Landmarks_full,Landmarks_baseline
from gent2_predictor.predictor.ffn_trainer import FFNTrainer
from gent2_predictor.predictor.transformer_trainer import TransformerTrainer
from gent2_predictor.settings import create_pathname, MODEL_SELECTOR, USE_FULL_DATA


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
        parser = DataParser()
        parser.parse_structures()
        parser.pickle_data()

    elif args.ffn_train:
        if USE_FULL_DATA == True:
            if MODEL_SELECTOR == 'FULL_FFN':
                model = FFN()
                trainer = FFNTrainer(model, USE_FULL_DATA)
                trainer.start_loop()
            else:
                model = Baseline_FFN()
                trainer = FFNTrainer(model, USE_FULL_DATA)
                trainer.start_loop()
        else:
            if MODEL_SELECTOR == 'FULL_FFN':
                model = Landmarks_full()
                trainer = FFNTrainer(model, USE_FULL_DATA)
                trainer.start_loop()
            else:
                model = Landmarks_baseline()
                trainer = FFNTrainer(model, USE_FULL_DATA)
                trainer.start_loop()


    elif args.predict_on_ffn:
        model_filename, model_path = create_pathname()
        if USE_FULL_DATA == True:
            if model_filename.startswith('b'):
                model = Baseline_FFN()
                model.load_state_dict(torch.load(model_path))
                trainer = FFNTrainer(model, USE_FULL_DATA)
                scores = trainer.predict(model_filename)

            else:
                model = FFN()
                model.load_state_dict(torch.load(model_path))
                trainer = FFNTrainer(model, USE_FULL_DATA)
                scores = trainer.predict(model_filename)
        else:
            if model_filename.startswith('b'):
                model = Landmarks_baseline()
                model.load_state_dict(torch.load(model_path))
                trainer = FFNTrainer(model, USE_FULL_DATA)
                scores = trainer.predict(model_filename)

            else:
                model = Landmarks_full()
                model.load_state_dict(torch.load(model_path))
                trainer = FFNTrainer(model, USE_FULL_DATA)
                scores = trainer.predict(model_filename)


    elif args.transformer_train:
        fraction = 4
        model = nn.Transformer(
            d_model=21920 // fraction, nhead=2, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=100)
        trainer = TransformerTrainer(model)
        trainer.start_loop(fraction)


if __name__ == "__main__":
    main()
