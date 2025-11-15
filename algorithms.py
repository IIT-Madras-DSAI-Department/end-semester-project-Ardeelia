import pandas as pd
import numpy as np
import math
import random
from collections import Counter
from scipy.stats import mode


def dot(x,y):
    return np.dot(x, y)

def dist(x, y):
    return np.sum((x - y) ** 2)

class Softmax: 
    def __init__(self):
        self.theta = [0]
    def update(self,theta,X,y):
        y = np.array(y)
        theta = np.array(theta)
        grads = [np.zeros(len(X[0])) for _ in range(len(theta))]
        for i in range(len(X)):
            individuals = [dot(theta[j], X[i]) for j in range(len(theta))]
            mex = max(individuals)
            exp_values = [math.exp(individuals[j] - mex) for j in range(len(individuals))]
            total = sum(math.exp(ind - mex) for ind in individuals)
            for j in range(len(theta)):
                if(y[i] == j):
                    grads[j] -= X[i]
                grads[j] += X[i]*(exp_values[j]/total)
        for j in range(len(theta)):
            theta_j = theta[j].ravel()
            theta[j] = theta_j - (0.05*grads[j])
        return theta 
    def fit(self,X,y,mb_iters,main_iters):
        X = np.asarray(X, float)
        theta_vals = [[0 for j in range(len(X[0]))] for i in range(10)]
        size = len(X)//mb_iters
        for j in range(main_iters):
            for i in range(mb_iters):
                theta_vals = self.update(theta_vals, X[size*i:size*(i+1)] , y[size*i:size*(i+1)])
        self.theta = np.array(theta_vals)
        print('finished softmax training')
    
    def predict(self,X):
        preds = []
        X = np.array(X)
        self.theta = np.array(self.theta)
        for i in range(len(X)):
            prediction = np.dot(self.theta, X[i])
            preds.append(np.argmax(prediction))
        return preds




class KMeans:
    def __init__(self):
        self.centers = []
    def fit(self, X, y, k, iters):
        X = np.array(X)
        random_indices = np.random.choice(len(X), size=k, replace=False)
        cluster_centers = X[random_indices]
        for i in range(iters):
            print(f'doing iter {i+1}')
            points_assigned_to_cluster = [[] for j in range(k)]
            total_loss = 0
            for j in range(len(X)):
                min_cluster_center = None
                min_dist = math.inf
                for l in range(len(cluster_centers)):
                    if(dist(cluster_centers[l], X[j]) < min_dist):
                        min_cluster_center = l
                        min_dist = dist(cluster_centers[l], X[j])
                total_loss += min_dist
                points_assigned_to_cluster[min_cluster_center].append(j)
            print(f"finished assigning centers with loss {total_loss}")
            for l in range(len(cluster_centers)):
                cluster_centers[l] = X[points_assigned_to_cluster[l]].mean(axis = 0)
        cluster_assignments = [np.bincount(y[points_assigned_to_cluster[l]]).argmax() for l in range(len(cluster_centers))]
        self.centers = cluster_centers
        self.assign = cluster_assignments
    def predict(self, X):
        X = np.array(X)
        preds = []
        for i in range(len(X)):
            min_dist_center = None
            min_dist = math.inf
            for j in range(len(self.centers)):
                if(dist(self.centers[j],X[i]) < min_dist):
                    min_dist = dist(self.centers[j],X[i])
                    min_dist_center = j
            preds.append(self.assign[min_dist_center])
        return preds


def KNN(Xtrain,ytrain,Xval,k):
    Xval = np.array(Xval)
    Xtrain = np.array(Xtrain)
    preds = []
    for i in range(len(Xval)):
        dist = np.linalg.norm(Xtrain - Xval[i], axis=1)
        idx = np.argpartition(dist, k)[:k]
        points = ytrain[idx]
        preds.append(np.bincount(points).argmax())
    return preds



class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_indices = feature_indices

    def fit(self, X, y):
        if not self.feature_indices:
            self.feature_indices = list(np.arange(X.shape[1]))
            assert len(self.feature_indices) > 1, 'Less than 2 features, tree building may fail'
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_classes = len(set(y))
        num_samples = len(y)

        if num_samples == 0:
            return None
        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None
        for feat_idx in self.feature_indices:
            if feat_idx >= X.shape[1]:
                continue
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) > 20:
                thresholds = np.random.choice(thresholds, 20, replace=False)
            for thresh in thresholds:
                gain = self._gini_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
        return split_idx, split_thresh


    def _gini_gain(self, y, feature_column, threshold):
        parent_gini = self._gini(y)
        left_idx = feature_column <= threshold
        right_idx = ~left_idx
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0
        n = len(y)
        n_left, n_right = np.sum(left_idx), np.sum(right_idx)
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return parent_gini - child_gini

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.is_leaf_node():
            return node.value
        if node.feature_index is None or node.feature_index >= len(inputs):
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)


class RandomForest:
    def __init__(self, n_trees=7, max_depth=5, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.max_features = self.max_features or X.shape[1]

        for tci in range(self.n_trees):
            print('~~~~~~~~~~~~~~~~~~~~~~~')
            print('Building tree', tci)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = random.sample(range(X.shape[1]), min(self.max_features, X.shape[1]))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices=feature_indices
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            print('Completed building tree', tci)
            print('~~~~~~~~~~~~~~~~~~~~~~~')

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        sample_size = int(0.1 * n_samples)
        indices = np.random.randint(0, n_samples, sample_size)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return final_preds

import numpy as np

def loo_knn_fast(Xtrain, ytrain, k=5):
    Xtrain = Xtrain.astype(float)
    ytrain = np.array(ytrain)
    n = len(Xtrain)
    pairwise_distances = np.sum(Xtrain**2, axis=1, keepdims=True) + \
            np.sum(Xtrain**2, axis=1) - 2 * Xtrain @ Xtrain.T
    for i in range(n):
        pairwise_distances[i][i] = np.inf

    knn_idx = np.argpartition(pairwise_distances, k, axis=1)[:, :k]
    
    predictions = [0]*n

    for i in range(n):
        neighbor_labels = ytrain[knn_idx[i]]
        predictions[i] = mode(neighbor_labels, keepdims=False).mode
    return predictions



import time 
def new_classes(Xtrain, ytrain):
    Xtrain = np.array(Xtrain).astype(float)
    ytrain = np.array(ytrain)
    Model1 = Softmax()
    Model1.fit(Xtrain, ytrain, 100, 100)
    pred1 = Model1.predict(Xtrain)
    Model2 = KMeans()
    Model2.fit(Xtrain, ytrain, 300, 5)
    pred2 = Model2.predict(Xtrain)
    start = time.time()
    pred3 = loo_knn_fast(Xtrain, ytrain, k=5)
    print(time.time() - start)
    final_class = []
    for i in range(len(ytrain)):
        correct_preds = []
        if pred3[i] == ytrain[i]:
            final_class.append(3)
        elif pred2[i] == ytrain[i]:
            final_class.append(2)
        elif pred1[i] == ytrain[i]:
            final_class.append(1)
        else:
            final_class.append(3)
    return np.array(final_class), Model1, Model2
