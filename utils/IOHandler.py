import os
import warnings
import time
import shutil
import re
import json
import pickle


def mkdir(dir_name, location='', overwrite=False):
    """
    Creates directory at given location with name dir_name
    If no location is precised, created in working directory
    Default setting don't allow overwriting if directory already exists
    """
    path_to_dir = location + "/" + dir_name
    if os.path.exists(path_to_dir):
        if overwrite:
            shutil.rmtree(path_to_dir)
            os.mkdir(path_to_dir)
        else:
            warnings.warn(f"directory {dir_name} already exists")
    else:
        os.mkdir(path_to_dir)


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


def write_file_name(root, suffix, params={}, timestamp=False):
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
