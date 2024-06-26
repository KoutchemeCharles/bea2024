import errno
import os, json
from pathlib import Path
from shutil import rmtree
import time
from dotmap import DotMap

def json2data(filename):
    """ loads a json file into a list (od dictionaries) """
    with open(filename,'r') as json_file:
        data = json.load(json_file)
    return data

def create_dir(path, clear=False):
    """ Creates a directory on disk. """
    if clear and os.path.exists(path):
        rmtree(path)
    
    try:
        Path(path).mkdir(parents=True, exist_ok=not clear) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    

def save(file, text):
    """ Saves a text in a text file. """
    with open(file, 'w') as fp:
        fp.write(str(text))

def save_json(data, filename):
    """ Saves data as json. """
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4)

def write(file, string):
    with open(file, "w") as fp:
        fp.write(string)

def read_config(filename):
    """ Read a dictionary in a configuration format and transforms it into DotMap."""
    return DotMap(json2data(filename))