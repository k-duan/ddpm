import unittest

import torch

from model import DDPM

class DDPMUnitTests(unittest.TestCase):
    def setUp(self):
        self._max_t = 100
        self._ddpm = DDPM(max_t=self._max_t, n_channels=1)

    def testBeta(self):
        expected_beta_t = [0.0001, 0.0003, 0.0039, 0.02]
        beta_t = [self._ddpm.beta_t(torch.tensor(t)).item() for t in [0, 1, 19, 99]]
        for b, expected_b in zip(beta_t, expected_beta_t):
            self.assertAlmostEqual(b, expected_b, delta=1e-4)

    def testAlphaBar(self):
        expected_alpha_bar_t = [0.9999, 0.9999*0.9997, 0.9999*0.9997*0.9995]
        # alpha_bar_0 = (1-beta_0)
        # alpha_bar_1 = (1-beta_0) * (1-beta1)
        # alpha_bar_2 = (1-beta_0) * (1-beta1) * (1-beta2)
        alpha_bar_t = [self._ddpm.alpha_bar_t(torch.tensor(t)).item() for t in [0, 1, 2]]
        for a, expected_a in zip(alpha_bar_t, expected_alpha_bar_t):
            self.assertAlmostEqual(a, expected_a, delta=1e-4)
