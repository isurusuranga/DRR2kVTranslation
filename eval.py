# this is for the test dataset
from options import TestOptions
from test import Evaluator

if __name__ == '__main__':
    options = TestOptions().parse_args()
    evaluator = Evaluator(options)
    evaluator.evaluate()

