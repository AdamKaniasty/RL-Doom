import vizdoom as vzd
import os
import argparse

from src.models.CustomPPO import CustomPPO_Model

CONFIG_PATH = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")

if __name__ == '__main__':
    model = CustomPPO_Model(
        config_file=CONFIG_PATH,
        mode='test',
        pretrained='./src/models/weights/WEIGHTS_PPO_DEFEND_transfer_basic_200000'
    )

    model.test()
