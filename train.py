from options import TrainOptions
from train import Trainer
from utils import *


if __name__ == '__main__':
    seed_all(42)
    options = TrainOptions().parse_args()
    print('start training.......')
    trainer = Trainer(options)
    trainer.model_train()
