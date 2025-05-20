################################   Libraries   ################################

import json
import os
from pathlib import Path

import cv2
import numpy as np

#############################   Global variables   ############################

BASE_DIR = Path(__file__).resolve().parent
np_key = None
sources = None
loading_screen_data = None


def get_absolute_path(path_str: str, from_cwd: bool = False) -> str:
    """
    Get the absolute path; from the path from root.

    Args:
        path_str (str): path of the file from the root folder
        from_cwd (bool): whether to get the path from current working directory
    Returns:
        str: absolute path of the file
    """
    if from_cwd:
        return os.path.join(os.getcwd(), path_str)
    return os.path.join(str(BASE_DIR), path_str)


def validate_settings_file():
    """
    Get the data from the settings file

    Returns:
        dict: Welcome messages from settings file
    """
    global sources, np_key

    settings_path = get_absolute_path(os.path.join('assets', 'configs', 'settings.json'))
    try:
        settings_file = open(settings_path)
        data = json.load(settings_file)
        return data
    except FileNotFoundError:
        print(f"Error: Unable to load file {settings_path}")
    except json.decoder.JSONDecodeError:
        print(f"Error: Corrupt file {settings_path}")
    except AssertionError:
        print(f"Error: Corrupt license key.... Please do not try to remove the author credits.")


def load_settings():
    """
    Load the required data from source file

    Returns:
        None
    """
    global sources, loading_screen_data

    data = validate_settings_file()
    sources = data["source"] if "source" in data else 0
    sources["FILE"] = get_absolute_path(sources["FILE"].replace("/", os.sep))
    loading_screen_data = data["welcome"]


def get_source(source_type):
    """
    Get the source path from settings file.

    Args:
        source_type (str): type of source to load
    Returns:
        str: source path
    """
    settings = validate_settings_file()
    if settings:
        return settings["source"][source_type]
    return 0  # Default to first webcam if settings not found


def load_files():
    """
    Load the required files into memory.

    Returns:
        tuple: cascade classifier & jewellery data
    """
    # Load the cascade file
    cascade_path = get_absolute_path(os.path.join('assets', 'model', 'haarcascade_frontalface_default.xml'))
    cascade = cv2.CascadeClassifier(cascade_path)

    # Load jewellery data
    jewel_path = get_absolute_path(os.path.join('assets', 'configs', 'jewellery.json'))
    try:
        jewel_file = open(jewel_path)
        jewelleries = json.load(jewel_file)

        # Update paths to use proper separators
        for key in jewelleries:
            jewelleries[key]["path"] = cv2.imread(
                get_absolute_path(jewelleries[key]["path"]),
                cv2.IMREAD_UNCHANGED
            )

        return cascade, jewelleries
    except FileNotFoundError:
        print(f"Error: Unable to load file {jewel_path}")
    except json.decoder.JSONDecodeError:
        print(f"Error: Corrupt file {jewel_path}")
    return None, None


def generate_loading_screen(height: int, width: int):
    """
    Generate the loading screen to show till the files are loading

    Args:
        height (int): height of the frame
        width (int): width of the frame
    Returns:
         np.array: loading screen as frame
    """
    loading_screen = np.zeros((height, width, 3))
    if not loading_screen_data:
        load_settings()
    return loading_screen