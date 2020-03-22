"""Sets paths based on configuration files."""

import configparser
import os
import types

_FILENAME = None
_PARAM = {}
for filename in ["echonet.cfg",
                 ".echonet.cfg",
                 os.path.expanduser("~/echonet.cfg"),
                 os.path.expanduser("~/.echonet.cfg"),
                 ]:
    if os.path.isfile(filename):
        _FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            _PARAM = config["config"]
        break

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATA_DIR=_PARAM.get("data_dir", "a4c-video-dir/"))
