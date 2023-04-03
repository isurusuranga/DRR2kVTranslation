import os
import json
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super(TestOptions, self).__init__()
        # pass dataroot argument to the image folder path which we need to transfer the style.
        # It should be either train or validation or test image folder path
        # pass test_results_dir argument to either train or validation or test directory to save the style transferred
        # images
        self.parser.add_argument('--test_results_dir', required=True, help='Directory to store test results')

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()

        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')

        self.save_dump()

        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "test_config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return


