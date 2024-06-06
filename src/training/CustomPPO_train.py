import vizdoom as vzd
import os
import argparse

from src.models.CustomPPO import CustomPPO_Model

CONFIG_PATH = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Custom PPO model.')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train the model')

    args = parser.parse_args()
    epochs = 100000

    args.epochs = epochs

    model = CustomPPO_Model(
        config_file=CONFIG_PATH,
        mode='train',
        pretrained='./src/models/weights/WEIGHTS_PPO_DEFEND_transfer_basic_100000'
    )
    model.train(args.epochs)
    model.save(f'WEIGHTS_PPO_DEFEND_transfer_basic_200000')
