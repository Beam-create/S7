import unittest

import numpy as np

import sys
sys.path.append('C:\\Users\\Christopher\\Desktop\\Workspace_temp\\S7\\APP1\\Problematique\\Problematique')


from dnn_framework import SgdOptimizer
from app1_tests import DELTA


class SgdOptimizerTestCase(unittest.TestCase):
    def test_step(self):
        parameters = {
            'x': np.array([10.0]),
            'y': np.array([10.0])
        }
        optimizer = SgdOptimizer(parameters, learning_rate=0.1)

        for i in range(100):
            parameter_grads = {
                'x': 2 * parameters['x'] + 1,  # f(x) = x^2 + x
                'y': 2 * parameters['y']  # f(y) = y^2
            }
            optimizer.step(parameter_grads)

        self.assertAlmostEqual(parameters['x'][0], -0.5, delta=DELTA)
        self.assertAlmostEqual(parameters['y'][0], 0.0, delta=DELTA)


if __name__ == '__main__':
    unittest.main()
