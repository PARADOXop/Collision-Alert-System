from collision_alert import logger
import sys
import os
import json
from ensure import ensure_annotations
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError



@ensure_annotations
def create_directories(path_to_dir, verbose = True):
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def read_yaml(path_to_yaml):
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e