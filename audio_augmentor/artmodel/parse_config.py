# The template parse_config.py file for loading the LCNN model.
# Copy this file to the same directory as the current code file.
# And import it but never use it.
import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
import logging.config
import json
from collections import OrderedDict

def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
        
def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class ConfigParser:
    def __init__(self, config: dict, options='', timestamp=True):
        # parse default and custom cli options


        # load config file and apply custom cli options
        self.__config = config
        self.__save_dir = None
        self.__log_dir = None


    def initialize(self, name, module):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_cfg = self[name]
        return getattr(module, module_cfg['type'])(**module_cfg['args'])

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self.__config

    @property
    def save_dir(self):
        return self.__save_dir

    @property
    def log_dir(self):
        return self.__log_dir
