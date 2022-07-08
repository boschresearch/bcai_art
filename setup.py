#!/usr/bin/env python
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

import os
from os import path
from setuptools import setup, find_packages
import unittest
import sys
import subprocess

ROOT_DIR_NAME = 'bcai_art'

sys.path.append(ROOT_DIR_NAME)
from version import __version__

print('Building version:', __version__)

curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

EXCLUDE_DIRS = ''.split()

package_list=[l.strip() for l in open('requirements.txt') if not l.startswith('#') and not l.startswith('git+') and l.strip() != '']

print(package_list)

setup(
    name=ROOT_DIR_NAME,
    version=__version__,
    description='BCAI ART: Bosch Center for Artificial Intelligence Adversarial Robustness Toolkit',
    python_requires='>=3.6',
    author="Robert Bosch GMBH",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # We want to distribute source code as well so the code could read, e.g., config_schema.json
    # see https://setuptools.readthedocs.io/en/latest/userguide/miscellaneous.html#setting-the-zip-safe-flag
    zip_safe=False,
    scripts=['bcai_art_run.py', 'bcai_art_run_autoattack.py'],
    packages=find_packages(where='.', exclude=EXCLUDE_DIRS),
    package_data={ROOT_DIR_NAME: ['config_schema.json']},
    setup_requires=package_list,
    install_requires=package_list
)


INSTALL_ADD_SCRIPT='./install_autoattack.sh'
try:
    print(subprocess.check_output([INSTALL_ADD_SCRIPT]).decode())
except:
    print(f'The build script {INSTALL_ADD_SCRIPT} failed')
    sys.exit(1)

# setup tools fail to find these tests (when one specifies test_suit arg) and it's not clear why
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(start_dir='./tests', pattern='test*.py')
test_runner = unittest.TextTestRunner()
results : unittest.runner.TextTestResult = test_runner.run(test_suite)
assert not results.errors
assert not results.failures
