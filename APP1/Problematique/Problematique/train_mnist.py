import argparse
import time

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=False, default="APP1\Problematique\Problematique\output_folder")

    parser.add_argument('--checkpoint_path', type=str, help='Choose the output path', default=None)

    args = parser.parse_args()
    start_time = time.time()
    network = create_network(args.checkpoint_path)
    trainer = MnistTrainer(network, args.learning_rate, args.epoch_count, args.batch_size, args.output_path)
    trainer.train()
    end_time = time.time()
    print("Temps d'exécution : ", end_time - start_time)


def create_network(checkpoint_path):
    layers = [FullyConnectedLayer(784, 128), 
              BatchNormalization(128, 0.1),
              ReLU(),
              FullyConnectedLayer(128, 32),
              BatchNormalization(32, 0.1),
              ReLU(),
              FullyConnectedLayer(32, 10),]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    main()
