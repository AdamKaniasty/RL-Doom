import vizdoom as vzd
import os

from src.models.CustomA2C import CustomA2C_Model
import argparse

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the A2C model.')
    parser.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train the model')

    args = parser.parse_args()

    model = CustomA2C_Model(
        CONFIG_PATH,
        mode='train'
    )
    # 5K -> around 8:30 minutes
    model.train(500_000)
    model.save('WEIGHTS_A2C_1_EXTENDED_500K.ZIP')
