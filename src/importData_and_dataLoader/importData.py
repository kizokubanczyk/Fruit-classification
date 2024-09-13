import os

def importData(file_path_data):

    fruit = []

    for dirname, _, filenames in os.walk(file_path_data):
        folder_name = os.path.basename(dirname)

        if folder_name and folder_name not in fruit_images:
            fruit_images[folder_name] = []

        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            image = cv2.imread(image_path)

            if image is not None:
                fruit_images[folder_name].append(image)
            else:
                fruits_failed_to_load += 1