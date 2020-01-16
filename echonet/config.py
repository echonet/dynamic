import os
import configparser
import types

FILENAME = None
param = {}
for filename in ["echonet.cfg",
                 ".echonet.cfg",
                 os.path.expanduser("~/echonet.cfg"),
                 os.path.expanduser("~/.echonet.cfg")]:
    if os.path.isfile(filename):
        FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            param = config["config"]
        break

config = types.SimpleNamespace(
        FILENAME=FILENAME,
        DATA_DIR=param.get("data_dir", "a4c-video-dir/"))
