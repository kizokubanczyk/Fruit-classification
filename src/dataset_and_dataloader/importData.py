from typing import List, Tuple
import os


def importData(data_path: str) ->Tuple[List[str], List[str]]:

    labels = []
    images = []
    folders_to_check = {'apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon'}

    for dirname, _, filenames in os.walk(data_path):
        folder_name = os.path.basename(dirname)

        if folder_name in folders_to_check:

            for filename in filenames:
                image_path = os.path.join(dirname, filename)

                images.append(image_path)
                labels.append(folder_name)

    return images, labels