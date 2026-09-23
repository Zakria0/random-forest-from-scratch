"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    p = dict()
    c = np.unique(labels)
    n = len(labels)
    for label in labels:
        p[label] = p.get(label, 0) + 1 / n
    
    return 1 - sum([p[key]**2 for key in p])

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    mask = features[:, feature_index] > threshold
    return features[~mask], labels[~mask], features[mask], labels[mask]

# Step 3 - split_score (not yet solved)
# TODO: implement

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

