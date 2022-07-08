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

import json
import jsonschema
import os
from types import SimpleNamespace

SCHEMA = os.path.join(os.path.dirname(__file__), "config_schema.json")

def dump_to_namespace(ns, d):
    for k, v in d.items():
        if isinstance(v, dict):
            leaf_ns = SimpleNamespace()
            ns.__dict__[k] = leaf_ns
            dump_to_namespace(leaf_ns, v)
        elif isinstance(v, list):
            arr = []
            ns.__dict__[k] = arr
            for e in v:
                if isinstance(e, dict):
                    leaf_ns = SimpleNamespace()
                    dump_to_namespace(leaf_ns, e)
                    arr.append(leaf_ns)
                else:
                    arr.append(e)
        else:
            ns.__dict__[k] = v

def _load_schema(filepath: str = SCHEMA) -> dict:
    with open(filepath, "r") as schema_file:
        schema = json.load(schema_file)
    return schema


def validate_settings(config: dict) -> dict:
    """
    Validates that a config matches the default JSON Schema
    """
    schema = _load_schema()

    jsonschema.validate(instance=config, schema=schema)
    config_ns = SimpleNamespace()
    dump_to_namespace(config_ns, config)  
    return config_ns


def load_settings(filepath: str) -> dict:
    """
    Loads and validates a config file
    """
    with open(filepath) as f:
        config = json.load(f)

    return validate_settings(config)
