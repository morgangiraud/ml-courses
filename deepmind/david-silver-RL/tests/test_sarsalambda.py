import os, sys, unittest
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from sarsalambda import getQLambda

class TestUtil(unittest.TestCase):

    def test_get_q_lambda(self):
        lambda_value = 0.9
        history = [
            0, { 'dealer': 1, 'user': 1, 'end': False }, 'hit',
            0, { 'dealer': 1, 'user': 2, 'end': False }, 'hit',
            1
        ]
        Qs = np.zeros([4, 2])
        current_step = 0
        q_lambda = getQLambda(lambda_value, history, Qs, current_step)
        self.assertEqual(q_lambda, 0.089999999999999983)

        history = [
            0, { 'dealer': 1, 'user': 1, 'end': False }, 'hit',
            0, { 'dealer': 1, 'user': 2, 'end': False }, 'hit',
            -1
        ]
        q_lambda = getQLambda(lambda_value, history, Qs, current_step)
        self.assertEqual(q_lambda, -0.089999999999999983)



if __name__ == '__main__':
    unittest.main()