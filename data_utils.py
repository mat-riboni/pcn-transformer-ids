import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    df = pd.read_csv(path) 
    X = df.drop(['Label', 'Attack'], axis=1) 
    y = df['Label']
    return X, y

def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def create_ssl_split(X_train, y_train, label_ratio=0.2): 
    X_unlabeled, X_labeled, y_unlabeled, y_labeled = train_test_split(
        X_train, y_train, test_size=label_ratio, stratify=y_train, random_state=42
    )
    return X_labeled, y_labeled, X_unlabeled