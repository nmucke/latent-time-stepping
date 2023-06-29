
import os

def create_directory(directory):
    """
    Creates a directory if it doesn't exist
    :param directory: The directory to create
    :return: None
    """

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)