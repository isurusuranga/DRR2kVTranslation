import os
import json
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super(TrainOptions, self).__init__()

        self.parser.add_argument('--name', type=str, default='drr2kv_experiment',
                            help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--resume', dest='resume', default=True, action='store_true',
                            help='Resume from checkpoint (Use latest checkpoint by default')
        self.parser.add_argument('--log_dir', default='logs', help='Directory to store logs')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='Total number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--ilw', type=float, default=5.0, help='Weight balance scalar for identity loss term')
        self.parser.add_argument('--clw', type=float, default=10.0,
                                 help='Weight balance scalar for cycle-consistency loss term')
        self.parser.add_argument('--decay_epochs', type=int, default=50, help='weight decay after every set of epochs')

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()

        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.args.output_dir = os.path.join(self.args.log_dir, 'outputs')
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.save_dump()

        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "train_config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return


