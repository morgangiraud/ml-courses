import os, sys, unittest
import numpy as np
from collections import Counter

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

import easy21

class TestUtil(unittest.TestCase):

    def test_draw(self):
        val_cnt = Counter()
        color_cnt = Counter()
        for i in range(5000):
            card = easy21.draw()
            val_cnt[card['value']] +=1
            color_cnt[card['color']] +=1

        
        self.assertEqual(color_cnt['red'] / 5000 < 1/3 + 0.05, True)
        self.assertEqual(color_cnt['red'] / 5000 > 1/3 - 0.05, True)
        self.assertEqual(val_cnt[1] / 5000 < 1/10 + 0.05, True)
        self.assertEqual(val_cnt[1] / 5000 > 1/10 - 0.05, True)
        self.assertEqual(val_cnt[10] / 5000 < 1/10 + 0.05, True)
        self.assertEqual(val_cnt[10] / 5000 > 1/10 - 0.05, True)

    def test_step(self):        
        current_state, reward = easy21.step()
        self.assertEqual(current_state['end'], False)
        self.assertEqual(reward, 0)

        current_state, reward = easy21.step(current_state, 'stick')
        self.assertEqual(current_state['end'], True)

    def test_get_state_id(self):        
        state = { 'dealer': 1, 'user': 1, 'end': False }
        stateId = easy21.getStateId(state)
        self.assertEqual(stateId, 0)

        state = { 'dealer': 10, 'user': 21, 'end': False }
        stateId = easy21.getStateId(state)
        self.assertEqual(stateId, 209)

    def test_get_action_id(self):        
        action = 'stick'
        actionId = easy21.getActionId(action)
        self.assertEqual(actionId, 0)

        action = 'hit'
        actionId = easy21.getActionId(action)
        self.assertEqual(actionId, 1)

    def test_phi(self):
        state = { 'dealer': 1, 'user': 1, 'end': False }
        action = 'hit'
        phi = easy21.phi(state, action)

        self.assertEqual(np.all(phi == [1] + [0  for i in range(35)]), True)

        action = 'stick'
        phi = easy21.phi(state, action)

        self.assertEqual(np.all(phi == [0, 1] + [0  for i in range(34)]), True)

        state = { 'dealer': 4, 'user': 4, 'end': False }
        action = 'hit'
        phi = easy21.phi(state, action)
        self.assertEqual(np.all(phi == [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,] + [0  for i in range(21)]), True)



if __name__ == '__main__':
    unittest.main()