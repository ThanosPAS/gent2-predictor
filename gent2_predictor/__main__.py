import argparse
import torch
from gent2_predictor.data_parser.data_parser import DataParser
from gent2_predictor.predictor import gent2-predictor

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

def main():
    parser = argparse.ArgumentParser(description="Deep learning package for Cancer type prediction from microarray gene expression data")

    parser.add_argument(
        '-p', '--parse', action='store_true',
        help='parse the data and pickle them to file')

    parser.add_argument(
        '-tf', '--ffn_train', action='store_true',
        help = "Train a fully connected deep neural network")

    parser.add_argument(
        '-pf', '--predict_on_ffn', action='store_true',
        help='Make predictions on the previously trained model')

    parser.add_argument(
        '-e', '--epochs', action='store_true', type = int, default = 100,
        help='Number of epochs for the model to train (Default: 100)')

    parser.add_argument(
        '-opt', '--optimizer', action='store_true', type=str, default='ADAM',
        help = 'Choose between SGD and ADAM (Default: ADAM)')

    parser.add_argument(
        '-lr', '--learning_rate', action='store_true', type = float, default = 0.001,
        help='The learning rate of the model (Default: 0.001)')

    parser.add_argument(
        '-reg', '--l2_regularization', action='store_true', type=int, default=0,
        help = "Choose optimizer's regularization value - either 0 or 0.1 (Default: 0)")

    parser.add_argument(
        '-mom', '--momentum', action='store_true', type = int, default = 0,
        help="Choose optimizer's momentum value - either 0 or 0.1 (Default: 0)")

    parser.add_argument(
        '-init', '--weight_bias_initialization', action='store_true', type = str, default = 'xavier_uniform',
        help='Weight and bias initialization method. Available methods: xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal, sparse. Default: xavier_uniform.')
    args = parser.parse_args()

    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    OPTIMIZER = args.optimizer
    INIT_METHOD = args.weight_bias_initialization
    L2_REG =args.l2_regularization
    MOMENTUM = args.momentum


    if args.parse:
        data_parser = DataParser()
        data = data_parser.parse_cancer_files()
        data_parser.pickle_data(data)
    elif args.tf:
        data_parser = DataParser()
        unpickled_patients = data_parser.unpickling()
        gent2-predictor.data_loading(unpickled_patients)

        # Weight initialization

        def init_weights(m):
            """
            https://pytorch.org/docs/master/nn.init.html
            """
            if isinstance(m, nn.Linear):
                if xavier_uniform:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)  # alternative command: m.bias.data.fill_(0.01)
                elif xavier_norm:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif kaiming_uniform:
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif kaiming_norm:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif orthogonal:
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif sparse:
                    nn.init.sparse_(m.weight)
                    nn.init.constant_(m.bias, 0)


        model = FFN()
        model.apply(init_weights)
        model = model.to(device)

        # Save model
        torch.save(model.state_dict(), FOLDER_PATH)

    elif args.pf:
        # Load model
        pretrained_model = FFN(*args, **kwargs)
        pretrained_model.load_state_dict(torch.load(FOLDER_PATH))
            # A path like this: '../data/p1ch2/mnist/mnist.pth'
        prediction = gent2-predictor.ffn_predict()



if __name__ == "__main__":
    main()
