import argparse


class BaseOptions(object):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataroot', required=True, help='path to images root folder')
        self.parser.add_argument('--img_res', type=int, default=256,
                                 help='Rescale images to size [img_res, img_res] before feeding to the network')
        self.parser.add_argument('--nf', type=int, default=64, help='number of filters for the first conv layer')

