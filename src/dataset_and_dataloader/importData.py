from typing import List, Tuple
import os


def importData(data_path: str) ->Tuple[List[str], List[str]]:

    labels = []
    image_directory_paths = []
    folders_to_check = {'apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon'}

    for dirname, _, filenames in os.walk(data_path):
        folder_name = os.path.basename(dirname)

        if folder_name in folders_to_check:

            for filename in filenames:
                image_path = os.path.join(dirname, filename)
                image_directory_paths.append(image_path)
                labels.append(folder_name)

    return image_directory_paths, labels

def importExternalData(data_path: str) -> Tuple[List[str], List[str]]:
    labels = []
    image_directory_paths = []

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:

            full_path = os.path.join(dirname, filename)
            image_directory_paths.append(full_path)

            label = os.path.splitext(filename)[0]
            labels.append(label)

    return image_directory_paths, labels

