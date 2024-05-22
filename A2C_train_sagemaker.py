import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from src.game.play import play
from src.game.train import train
from src.models.stable_baselines3_A2C import A2C_Model
import sagemaker
from sagemaker.pytorch import PyTorch

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

output_path = f's3://{sagemaker_session.default_bucket()}/vizdoom'
config_file = f's3://{sagemaker_session.default_bucket()}/vizdoom/deadly_corridor.cfg'
print(f"Output path: {output_path}")
print(f"Config file: {config_file}")

if __name__ == '__main__':
    model = A2C_Model(
        CONFIG_PATH,
        mode='train'
    )
    # 5K -> around 8:30 minutes
    model.train(30000)
    model.save('a2c_vizdoom_30K.zip')
