#!/usr/bin/env python
import unittest
import torch
import numpy as np
import numpy.testing as npt

import sys

from bcai_art.utils_tensor import project, get_start_delta, get_last_data_dim,\
                                    START_ZERO, START_RANDOM
from bcai_art.utils_misc import NORM_INF, get_norm_code

NORM_LIST = [1, 2, 3, 4, 5, np.inf]
TOL_EPS=1e-5

SAMPLE_QTY_SHAPE = [11, 987]


class TestConfig(unittest.TestCase):

    def test_norm_conf1(self):
        self.assertTrue(get_norm_code(None) is None)

    def test_norm_conf2(self):
        for i in range(1, 1000):
            self.assertEqual(i, get_norm_code(f'L{i}'))
            self.assertEqual(i, get_norm_code(f'L{i}'))

    def test_norm_conf3(self):
        self.assertEqual(get_norm_code(NORM_INF), np.inf)
        self.assertEqual(get_norm_code(NORM_INF.lower()), np.inf)

    def test_norm_invalid(self):
        self.assertRaises(Exception, get_norm_code, 'l0')
        self.assertRaises(Exception, get_norm_code, 'l-1')
        self.assertRaises(Exception, get_norm_code, 'a0')


class TestStart(unittest.TestCase):

    def test_zero_start(self):
        for is_matr in [True, False]:
            for d in [1, 2, 3, 4, 5, 6]:
                DIM=[d]
                if is_matr:
                    DIM.append(2)
                SHAPE = tuple(SAMPLE_QTY_SHAPE + DIM)
                template = torch.zeros(SHAPE)

                # A zero start should always return zero
                for flag in [False, True]:
                    y = get_start_delta(template, START_ZERO, 0, 1, flag, is_matr=is_matr)
                    self.assertEqual(tuple(y.shape), SHAPE)
                    self.assertTrue(torch.sum(torch.abs(y)).item() < 1e-10)
                    self.assertTrue(y.requires_grad == flag)


    def test_rand_start(self):
        for is_matr in [False, True]:
            for d in [1, 2, 3, 4, 5, 6]:
                DIM=[d]
                if is_matr:
                    DIM.append(2)
                EPS=1
                SHAPE = tuple(SAMPLE_QTY_SHAPE + DIM)
                template = torch.zeros(SHAPE)

                for norm in NORM_LIST:
                    for flag in [False, True]:
                        y = get_start_delta(template, START_RANDOM, EPS, norm, flag, is_matr=is_matr)
                        self.assertEqual(tuple(y.shape), SHAPE)
                        y_norm = torch.norm(y, p=norm, dim=get_last_data_dim(is_matr)).view(-1).detach().numpy()
                        self.assertTrue(y.requires_grad == flag)
                        self.assertTrue(np.max(y_norm) < EPS + TOL_EPS)
                        qty = len(y_norm)

                        # Now let's check that norm values are distributed more or less uniformly
                        BIN_QTY=20
                        bins=1/BIN_QTY * np.array(range(0, BIN_QTY + 1, 1))
                        dist = np.histogram(y_norm, bins=bins)[0]

                        # This ensures that each bin is neither too fat or too skinny
                        # Clearly bin number and fractions are hardcoded, if any tests
                        # parameters change, the expected bin band might need to change as well

                        for k in range(BIN_QTY):
                            self.assertTrue(dist[k] >= 0.7 * qty / BIN_QTY and dist[k] <= 1.3 * qty / BIN_QTY)


class TestProjection(unittest.TestCase):

    def test_proj1(self):
        torch.manual_seed(0)

        for is_matr in [False, True]:
            for d in [1, 2, 3, 4, 5, 6]:
                DIM=[d]
                if is_matr:
                    DIM.append(2)
                EPS=1
                SHAPE = tuple(SAMPLE_QTY_SHAPE + DIM)
                # A lot of these vectors will have norm > EPS
                # but not all of them
                x = EPS * 1.1 * torch.randn(SHAPE)

                check_small_norm_done = False

                for norm in NORM_LIST:
                    y = project(x, eps=EPS, norm_p=norm, is_matr=is_matr)
                    self.assertEqual(tuple(y.shape), SHAPE)
                    x_norm = torch.norm(x, p=norm, dim=get_last_data_dim(is_matr)).view(-1).numpy()
                    y_norm = torch.norm(y, p=norm, dim=get_last_data_dim(is_matr)).view(-1).numpy()
                    qty = len(x_norm)
                    # First let's check that none of the projected vectors is
                    # outside the ball
                    self.assertEqual(np.sum(y_norm < EPS + TOL_EPS), qty)
                    # Second, for vectors originally inside the ball,
                    # the contents should change minimally
                    idx = x_norm < EPS
                    # Hopefull, there are some vectors with contents inside the ball.
                    if np.sum(idx) == 0:
                        continue

                    diff = (x_norm[idx] - y_norm[idx])
                    diff = diff * diff

                    self.assertTrue(np.mean(diff) < TOL_EPS)
                    check_small_norm_done = True

                # A rather hacky way to ensure that test data is sufficient
                # to generate vectors that originally have norm < EPS,
                # so we could check their norms are unchanged after the projection!
                self.assertTrue(check_small_norm_done)



if __name__ == "__main__":
    unittest.main()
