from sklearn.model_selection import train_test_split
from typing import (Tuple, List)


def split(fruit_labels_images: dict) -> Tuple[List, List, List, List, List, List]:

    list_labels = []
    list_smaples = []

    for label, smaples in fruit_labels_images.items():
        for smaple in smaples:
            list_labels.append(label)
            list_smaples.append(smaple)

    X_train, X_test, y_train, y_test = train_test_split(list_smaples, list_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test
