from sklearn.model_selection import train_test_split
from typing import (Tuple, List)

def split(image_directory_paths: list[str], labels: list[str]) -> Tuple[List, List, List, List, List, List]:


    X_train, X_test, y_train, y_test = train_test_split(image_directory_paths, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test


