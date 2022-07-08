#
# BCAI ART : Bosch Center for AI Adversarial Robustness Toolkit
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import torch

from random import Random

class DataPartitioner(object):
    """
    Dataset partitioning helper, taken (with tiny modifications)
    from https://pytorch.org/tutorials/intermediate/dist_tuto.html
    """

    def __init__(self, data, size_fracs, seed):
        """Construct a series of data subsets.

        :param data:        the original data set.
        :param size_fracs:   partition sizes as fractions of the original size.
        :param seed:
        """
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in size_fracs:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return torch.utils.data.Subset(self.data, self.partitions[partition])
