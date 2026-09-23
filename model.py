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

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    l, r, p = len(left_labels), len(right_labels), len(parent_labels)
    il, ir, ip = impurity(left_labels), impurity(right_labels), impurity(parent_labels)
    return ip - (l / p * il + r / p * ir)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_fi, best_t, best_s = None, None, 0.0
    for feature_index in feature_indices:
        L = np.unique(features[:, feature_index])
        thresholds = [(L[i] + L[i+1]) / 2 for i in range(len(L)-1)]
        for threshold in thresholds:
            left_features, left_labels, right_features, right_labels = split_dataset(features, labels, feature_index, threshold)
            if not len(left_features) or not len(right_features):
                continue
            s = split_score(labels, left_labels, right_labels)
            if s > best_s:
                best_fi, best_t, best_s = feature_index, threshold, s
    
    
    
    return {
        'feature_index': best_fi, 
        'threshold': best_t, 
        'score': best_s
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if depth >= max_depth:
        return True
    if len(np.unique(labels)) == 1:
        return True
    if len(labels) < min_samples_split:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    classes, counts = np.unique(labels, return_counts=True)
    return int(classes[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    candidate_features = range(features.shape[1]) if feature_subset is None else list(feature_subset)
    
    split = best_split(features, labels, candidate_features)
    fi, t = split["feature_index"], split["threshold"]
    if fi is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    left_features, left_labels, right_features, right_labels = split_dataset(features, labels, fi, t)
    
    if not len(left_features) or not len(right_features):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    return {'leaf': False, 'feature_index': fi, 'threshold': t, 'left': build_tree(left_features, left_labels, depth=depth+1, feature_subset=feature_subset, max_depth=max_depth, min_samples_split=min_samples_split), 'right': build_tree(right_features, right_labels, depth=depth+1, feature_subset=feature_subset, max_depth=max_depth, min_samples_split=min_samples_split)}

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    if tree['leaf']:
        return tree['prediction']
    if example[tree['feature_index']] <= tree['threshold']:
        return predict_example_tree(tree['left'], example)
    else:
        return predict_example_tree(tree['right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    predictions = []
    for row in features:
        predictions.append(predict_example_tree(tree, row))

    return np.array(predictions)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    n = labels.shape[0]
    idx = rng.integers(0, n, size=n) 
    return features[idx], labels[idx]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.permutation(num_features)[:num_to_pick]

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    n_samples, n_features = features.shape

    forest = []
    rng = np.random.default_rng(random_state)
    if num_features_per_split is None:
        num_features_per_split = max(1, int(np.round(np.sqrt(n_features))))
    
    for _ in range(num_trees):
        sample_features, sample_labels = bootstrap_sample(features, labels, rng=rng)
        feature_idx = feature_subset(n_features, num_features_per_split, rng=rng)

        tree = build_tree(sample_features, sample_labels, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feature_idx, depth=0)

        forest.append({'tree': tree, 'feature_indices': feature_idx})
    
    return forest

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

