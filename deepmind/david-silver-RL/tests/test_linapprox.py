import os, sys, unittest
import numpy as np
from collections import Counter

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

import linapprox

class TestUtil(unittest.TestCase):
    # dummy
    lambda: 1


if __name__ == '__main__':
    unittest.main()