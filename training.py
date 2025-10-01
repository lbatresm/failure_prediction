"""Script to load the data prepared with 'trainind_data_prep.py'"""

import pandas as pd
from pathlib import Path
import helpers as hp
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def _quick_verifications (df_chan: pd.DataFrame, df_chips: pd.DataFrame, df_labeled: pd.DataFrame):
    assert "chip_uid" in df_chan.columns, "chip_uid column is missing in df_a"
    assert "chip_uid" in df_chips.columns, "chip_uid column is missing in df_b"
    assert "chip_uid" in df_labeled.columns, "chip_uid column is missing in df_labeled"
    #TODO: Verify that all of the dataframes have the same chip:uids

def _prepare_chip_training (df_chip: pd.DataFrame, df_labeled: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Add labels to chip df
    df_chips = df_chip.merge(df_labeled, on="chip_uid")
    print(f"Training will be performed on {len(df_chips)} chips")

    # Select chip features
    identif_cols = ("chip_uid", "label")
    feature_cols = [c for c in df_chips.columns if c not in identif_cols]

    # Select training variables (X and Y)
    """
    In ML we have 2 training variables: X,Y.
    X: features, independent variables. The model uses them to learn.
    It does not contain the target variable.
    
    Y: target independent variable, the label. The result we want to predict.
    """
    x = df_chips[feature_cols]
    y = df_chips["label"].astype(str).copy()

    # Encode labels
    """ML algorithms can't work with text as target (y). Convert to numbers"""
    y_enc = LabelEncoder().fit_transform(y)

    return x, y, y_enc

def _train_model (x, y, y_enc):

    # Define Stratified Cross Validation
    """
    'StratifiedKFold' We divide the dataset in k parts (folds). We train k-1 parts, and test the model with the
    remaining one. 'Stratified' means that you mantain the same proportion of classes (eg. good, missingcch, bad...)
    than in the full dataset. We use this when you have unbalanced classes (eg. lot of good, some empty).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define classifier model (RandomForest)
    clf = RandomForestClassifier(
        n_estimators=400, # Number of trees in the forest.
        max_depth=None, # Max depth (ramifications) of each tree
        min_samples_leaf=3, # Each terminal node (leaf) must have at least 3 samples. Avoids the tree from creating too specific rules (overfitting)
        n_jobs=-1, # Uses all CPU nodes in parallel
        class_weight="balanced", # Gives more weight to minority classes (labels)
        random_state=42 # Seed to ensure repeatable results (42 is a common value)
    )

    # Start cross validation
    """
    We'll use as evaluation metric the F1_macro (mean of F1 score for each class)"
    It's better than 'accuracy' if classes are unbalanced
    """
    scores = cross_val_score(clf, x, y_enc, cv=cv, scoring="f1_macro")

    # Print scores of cross validation
    """Mean: model mean performance. Std: how consistent is performance between folds"""
    print(f"F1_macro CV: mean={scores.mean():.3f} ± {scores.std():.3f}")

    # Train model on labeled data
    clf.fit(x, y_enc)


def main():

    # Load data
    channel_training = hp.load_file("a_channel_training_data.csv", "\t")
    chip_training = hp.load_file("b_chip_training_data.csv", "\t")
    labeled = hp.load_file("labeled_y_n.txt", "\t")

    # Verifications
    _quick_verifications(channel_training, chip_training, labeled)

    # Training preparation
    x, y, y_enc = _prepare_chip_training(chip_training, labeled)
    _train_model(x, y, y_enc)

    print("Workflow finished!")





if __name__ == '__main__':
    main()
