import vizdoom as vzd
import os
import argparse

from src.models.CustomPPO import CustomPPO_Model

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Custom PPO model.')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train the model')

    args = parser.parse_args()
    iter = 0
    model_name = 'WEIGHTS_PPO_5_FINALREWARD_600K'
    model = CustomPPO_Model(
        config_file=CONFIG_PATH,
        mode='test',
        # pretrained='./src/models/weights/' + model_name
    )
    model.train(args.epochs)
    model.save(f'WEIGHTS_PPO_5_FINALREWARD_100K')
