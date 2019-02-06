import os
import warnings
import time
import shutil
import re
import json
import pickle


def mkdir(dirname, location='.', overwrite=False, timestamp=False):
    """
    Creates directory at given location with name dir_name
    If no location is precised, created in working directory
    Default setting don't allow overwriting if directory already exists
    """
    if timestamp:
        dirname = dirname + "_" + time.strftime("%Y%m%d-%H%M%S")
    full_path = os.path.join(location, dirname)
    if os.path.exists(full_path):
        if overwrite:
            shutil.rmtree(full_path)
            os.mkdir(full_path)
        else:
            warnings.warn(f"directory {full_path} already exists")
    else:
        os.mkdir(full_path)


def dict_to_list(object, key="", paramsList=[]):
    """
    Takes dictionnary as an input and returns a list
    expliciting all the values of the dictionnary
    """
    if isinstance(object, dict) and len(object.keys()) > 0:
        keySet = object.keys()
        for key in keySet:
            dict_to_list(object[key], key, paramsList=paramsList)
    else:
        str_param = str(key) + str(object)
        paramsList += [str_param]
        paramsList.sort()
    return paramsList


def dict_to_string(object, key="", paramsList=[]):
    """
    Takes dictionnary as an input and returns a string
    expliciting all the values of the dictionnary
    """
    paramsList = dict_to_list(object, paramsList=[])
    return "_".join(paramsList)


def write_file_name(root, suffix="", params={}, timestamp=False):
    """
    Generates saving name for a file
    Name format : root_param1Value1_param2Value2_param3Value3_..._timestamp.suffix
    """
    str_params = dict_to_string(params, paramsList=[])
    str_params = re.sub(" ", "", str_params)
    str_params = re.sub(",", "-", str_params)
    fileName = root + "_" + str_params
    if timestamp:
        fileName = fileName + "_" + time.strftime("%Y%m%d-%H%M%S")
    fileName = fileName + "." + suffix
    return fileName


def already_exists(path):
    """
    Returns True if file or directory already exists
    """
    return os.path.isfile(path)


def copy_file_to(src_path, tgt_dir_path, overwrite=False):
    """
    Copies specified file to directory
    """
    if not (already_exists(src_path) and already_exists(tgt_dir_path)):
        raise RuntimeError("Invalid specified path")
    else:
        tgt_path = os.path.join(tgt_dir_path, src_path)
        if already_exists(tgt_path) and not overwrite:
            raise RuntimeError("File already exists")
        else:
            shutil.copyfile(src_path, tgt_dir_path)


def save_json(path, jsonFile):
    """
    Dumps dictionnary as json file
    """
    with open(path, "wb") as f:
        f.write(json.dumps(jsonFile))


def load_json(path):
    """
    Loads json format file into python dictionnary
    """
    with open(path, "rb") as f:
        jsonFile = json.load(f)
    return jsonFile


def save_pickle(path, file):
    """
    Dumps file as pickle serialized file
    """
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    """
    Loads pickle serialized file
    """
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file
