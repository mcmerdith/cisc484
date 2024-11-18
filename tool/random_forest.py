from dataclasses import dataclass
import datetime
from multiprocessing import Pool
from typing import Literal, Self
import numpy as np
from sklearn.metrics import accuracy_score
from numpy.random import default_rng

DEBUG = False

UntrainedValueMode = Literal["majority", "nearest"]
"""How the model should handle unknown values

Options
-------
    `majority`
        return the most common label

    `nearest`
        return the label closest to the unknown value"""


def DEBUG_PRINT(*args, **kwargs):
    print("[DEBUG]", *args, **kwargs)


def WARN(*args, **kwargs):
    print("[WARN]", *args, **kwargs)


@dataclass
class BaseNode:
    feature_idx: int
    """The index of the feature this node predicts"""

    def count(self) -> int:
        raise NotImplementedError()

    def _get_branch(self, X: float, untrained_value_mode: UntrainedValueMode = None) -> Self | float | None:
        """Get the value of the branch for a sample.

        Must be implemented by subclasses if `predict` is not overridden.

        Parameters
        ----------
            `X` : int
                The value of the feature to predict the label for

            `untrained_value_mode` : UntrainedValueMode

        Returns
        -------
            A `BaseNode` or None if there is no suitable prediction"""
        raise NotImplementedError()

    def predict(self, sample: np.ndarray, untrained_value_mode: UntrainedValueMode = None) -> int:
        """Predict the label for a dataset

        Parameters
        ----------
            `sample` : np.ndarray<M>
                A sample of features

            `untrained_value_mode` : UntrainedValueMode
        """
        assert len(sample.shape) == 1, "sample must be a 1D array"

        predictor = self._get_branch(
            sample[self.feature_idx], untrained_value_mode)

        if predictor is None:
            raise RuntimeError("[PANIC] Failed to predict value of feature",
                               self.feature_idx, sample[self.feature_idx])

        if isinstance(predictor, BaseNode):
            return predictor.predict(sample, untrained_value_mode)
        else:
            return predictor


@dataclass
class LeafNode(BaseNode):
    """A leaf node in a decision tree

    The value of `feature_idx` is ignored"""
    label: float

    def count(self) -> int:
        return 1

    def predict(self, _: np.ndarray, __: UntrainedValueMode = None) -> float:
        return self.label


@dataclass
class BranchNode(BaseNode):
    branches: dict[float, BaseNode]
    majority_label: float

    def count(self):
        return sum([branch.count() for branch in self.branches.values()]) + 1

    def _get_branch(self, X: float, untrained_value_mode: UntrainedValueMode = None) -> BaseNode | float | None:
        branch = self.branches.get(X)
        if branch is None and untrained_value_mode is not None:
            if untrained_value_mode == "majority":
                if DEBUG:
                    DEBUG_PRINT("No branch found, returning majority label",
                                self.feature_idx, "[", X, "] =", self.majority_label)
                return self.majority_label
            elif untrained_value_mode == "nearest":
                keys = np.array(list(self.branches.keys()))
                nearest = keys[np.argmin(np.abs(keys - X))]
                if DEBUG:
                    DEBUG_PRINT("No branch found, returning nearest label",
                                self.feature_idx, "[", X, "] ->", nearest)
                return self.branches[nearest]
        return branch


@dataclass
class BinaryThresholdNode(BaseNode):
    threshold: float
    left: BaseNode
    right: BaseNode

    def count(self):
        return self.left.count() + self.right.count() + 1

    def _get_branch(self, X: float, _: UntrainedValueMode = None) -> BaseNode | float | None:
        if X < self.threshold:
            return self.left
        return self.right


############################
##### Helper Functions #####
############################


def _most_common(x: np.ndarray) -> int:
    """Return the most common value in `x`

    Parameters
    ----------
        `x` : np.ndarray
            An array of values
    """
    possible = np.unique(x)
    frequency = [len(x[x == value]) for value in possible]
    return possible[np.argmax(frequency)]


def _n_log_n(n: float) -> float:
    """Return the value of `n*log_2(n)`"""
    return n * np.log2(n)


############################
##### Information Gain #####
############################


def _entropy(n: np.ndarray) -> float:
    """Calculate the entropy of `n`

    Parameters
    ----------
        `n` : np.ndarray<N>
            A 1D array of values
    """

    log_sum = 0

    for label in np.unique(n):
        log_sum += _n_log_n(len(n[n == label]) / n.size)

    return -1 * log_sum


def _conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the entropy of `y` given `x`

    Parameters
    ----------
        `x` : np.ndarray<N>
            A 1D array of the feature values

        `y` : np.ndarray<N>
            A 1D array of the label values

    Restrictions
    ----------
        `x` and `y` must have the same size
    """
    assert x.size == y.size, "x and y must have the same size"

    conditional_entropy = 0
    for x_i in np.unique(x):
        # Get the set of labels associated with each value
        y_x = y[x == x_i]
        # Get the probability of the value
        p_x = (x[x == x_i]).size / x.size
        # Sum the logs of the probabilities of each label
        sum_y_x = 0
        for y_x_i in np.unique(y_x):
            sum_y_x += _n_log_n((y_x[y_x == y_x_i]).size / y_x.size)
        # Add to the total
        conditional_entropy += p_x * sum_y_x
    return -1 * conditional_entropy


def _information_gain(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Calculate the information gain of `x` given `y`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values

    Returns
    -------
        `information_gain` : list[float]<M>
            The information gain of each feature

    Restrictions
    ----------
        `N` and `y.size` must be equal
    """
    assert x.shape[0] == y.size, "x and y must have the same size"

    # Calculate entropy of y
    entropy_y = _entropy(y)

    # Calculate information gain of each feature
    information_gain = [
        entropy_y - _conditional_entropy(x[:, i], y)
        for i in range(x.shape[1])
    ]

    return information_gain


def _threshold_information_gain(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Calculate the information gain of `y` given `x` with `threshold`

    Parameters
    ----------
        `x` : np.ndarray<N>
            A 1D array containing `N` feature values

        `y` : np.ndarray<N>
            A 1D array of the label values

    Returns
    -------
        `information_gain` : float
            The information gain of `y` given `x`
    """
    # IG(Y|X:t) = H(Y)-H(Y|X:t)
    # H(Y|X:t) = P(X<t)*H(Y|X<t) + P(X>=t)*H(Y|X>=t)

    entropy_y = _entropy(y)

    # Segment the data
    xlt = x[x < threshold]
    yxlt = y[x < threshold]
    xgt = x[x >= threshold]
    yxgt = y[x >= threshold]

    # Calculate the probabilities of the segments
    p_xlt = xlt.size / x.size
    p_xgt = xgt.size / x.size

    # Calculate the entropy of the segments
    entropy_xlt = _conditional_entropy(xlt, yxlt)
    entropy_xgt = _conditional_entropy(xgt, yxgt)
    entropy_y_xt = p_xlt*entropy_xlt + p_xgt*entropy_xgt

    return entropy_y - entropy_y_xt


def _feature_set(x: np.ndarray, feature: int) -> np.ndarray:
    """Get the set of values for a feature

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `feature` : int
            The index of the feature to get the set of values for

    Returns
    -------
        `feature_set` : np.ndarray<N>
            The set of values for the feature
    """
    return x[:, feature]


############################
#####   Partitioning   #####
############################


def _real_value_split(x: np.ndarray, y: np.ndarray, feature_values: np.ndarray, min_samples: int = 2) -> \
        tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Split `x` and `y` into two sets based on `feature_values`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values

        `feature_values` : np.ndarray<N>
            The values of the feature to split on

        `min_samples` : int
            The minimum number of samples in each set to be considered

    Returns
    -------
        `threshold` : float
            The value to split on

        `left` : tuple[np.ndarray<N,M>, np.ndarray<N>]
            The left set of samples and labels

        `right` : tuple[np.ndarray<N,M>, np.ndarray<N>]
            The right set of samples and labels
    """

    # Initial settings
    best_threshold = None
    alt_threshold = None
    best_information_gain = -np.inf
    alt_information_gain = -np.inf
    best_imbalance = np.inf

    # Calculate possible thresholds
    thresholds = []
    sort_order = np.argsort(feature_values)
    sorted_features = feature_values[sort_order]
    sorted_y = y[sort_order]
    for i in range(sorted_features.size - 1):
        # Throw out thresholds that don't split the data
        if sorted_y[i] == sorted_y[i+1]:
            continue
        # Add the midpoint between the 2 features as a threshold
        thresholds.append(
            sorted_features[i] +
            ((sorted_features[i+1] - sorted_features[i]) / 2)
        )

    # Calculate the information gain for each threshold
    for threshold in np.unique(thresholds):
        information_gain = _threshold_information_gain(
            feature_values, y, threshold)

        # Get the segments
        xlt = feature_values < threshold
        xgt = feature_values >= threshold

        # Throw out thresholds that don't split the data
        if xlt.nonzero()[0].size < min_samples or \
                xgt.nonzero()[0].size < min_samples:
            continue

        # Calculate the imbalance of the segments
        imbalance = np.abs(xlt.nonzero()[0].size - xgt.nonzero()[0].size)

        # If information gain is improved save the threshold
        if information_gain > best_information_gain:
            best_threshold = threshold
            best_information_gain = information_gain

        # If imbalance is reduced save the threshold
        if imbalance < best_imbalance:
            alt_threshold = threshold
            alt_information_gain = information_gain
            best_imbalance = imbalance

    # Quick return if no threshold was found
    if best_threshold is None:
        return None

    # If the information gain is the same, use the threshold with the lowest imbalance
    if np.isclose(best_information_gain, alt_information_gain):
        best_threshold = alt_threshold

    # Get the segments
    xlt = feature_values < best_threshold
    xgt = feature_values >= best_threshold
    left_data = x[xlt, :]
    left_labels = y[xlt]
    right_data = x[xgt, :]
    right_labels = y[xgt]

    return best_threshold, (left_data, left_labels), (right_data, right_labels)


def _partition(x: np.ndarray, y: np.ndarray, feature_values: np.ndarray) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Partition `x` according to `feature_values`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values 

        `feature_values` : np.ndarray<N>
            The values of the feature to partition on

    Returns
    -------
        `partitions` : dict[float, tuple[np.ndarray<N,M>, np.ndarray<N>]]
            A dict of `S` partitions of `x` according to the feature of `x`
            with the highest information gain and the corresponding label values
    """

    # Partition the new array according to the value of the feature
    partitions = {
        feature_value: (
            x[feature_values == feature_value, :],
            y[feature_values == feature_value]
        ) for feature_value in np.unique(feature_values)
    }

    return partitions


def _create_node(x: np.ndarray, y: np.ndarray, *, binary_split_threshold: int = None, _include: np.ndarray = None) -> BaseNode:
    """Create a decision tree from `x` and `y`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values

    Returns
    -------
        `node` : BaseNode
            A decision tree node

    Restrictions
    ------------
        `N` and `y.size` must be equal
    """
    assert x.shape[0] == y.size, "x and y must have the same size"

    ##########################
    ###     Base Cases     ###
    ##########################

    # If there is only one label, return a leaf node
    unique_labels = np.unique(y)
    if unique_labels.size == 1:
        return LeafNode(feature_idx=None, label=unique_labels[0])

    # If there is only one set of inputs, return a leaf node
    unique_values = np.unique(x, axis=0)
    if len(unique_values) == 1:
        return LeafNode(feature_idx=None, label=_most_common(y))

    ##########################
    ### Recursive Behavior ###
    ##########################

    # Select the best feature to branch on
    if _include is None:
        if DEBUG:
            DEBUG_PRINT("Using all features")
        # Create an include array with all features
        _next_include = np.array([True] * x.shape[1])
        feature_idx = np.argmax(_information_gain(x, y))
    else:
        if DEBUG:
            DEBUG_PRINT("Using features", _include)
        # Copy the include array so mutations don't affect the original
        _next_include = _include.copy()
        # Get the feature index with the highest information gain
        feature_idx = np.argmax(_information_gain(x[:, _next_include], y))
        # Convert the feature index to the original index
        feature_idx = _next_include.nonzero()[0][feature_idx]

    if DEBUG:
        DEBUG_PRINT("Branching on feature", feature_idx)

    # Get the set of values for the feature
    feature_set = _feature_set(x, feature_idx)

    if binary_split_threshold is not None and feature_set.size > binary_split_threshold:
        # Real value split (binary tree)
        # The feature is not excluded from the next iteration as it may get split again

        value_split = _real_value_split(x, y, feature_set)

        if value_split is None:
            if DEBUG:
                WARN("Failed to create real value split")

            # return LeafNode(feature_idx=feature_idx, label=_most_common(y))
        else:
            threshold, \
                (left_data, left_labels), \
                (right_data, right_labels) \
                = value_split

            if DEBUG:
                DEBUG_PRINT("Creating real value split @ [", threshold, "]")
                DEBUG_PRINT("Left", left_data.shape[0], "samples and",
                            left_labels.shape[0], "labels")
                DEBUG_PRINT("Right", right_data.shape[0], "samples and",
                            right_labels.shape[0], "labels")

            return BinaryThresholdNode(
                feature_idx=feature_idx,
                threshold=threshold,
                left=_create_node(left_data, left_labels,
                                  binary_split_threshold=binary_split_threshold, _include=_next_include),
                right=_create_node(right_data, right_labels,
                                   binary_split_threshold=binary_split_threshold, _include=_next_include)
            )

    # Create a standard node
    # The feature is excluded from the next iteration
    _next_include[feature_idx] = False

    partitions = _partition(x, y, feature_set)
    nodes: dict[float, BaseNode] = {}

    if DEBUG:
        DEBUG_PRINT("Creating", len(partitions), "partitions")
        for feature_value, (p_data, p_labels) in partitions.items():
            DEBUG_PRINT("Partition", feature_value, "contains",
                        p_data.shape[0], "samples and", p_labels.shape[0], "labels")

    # Create the nodes for each partition
    for feature_value, (p_data, p_labels) in partitions.items():
        nodes[feature_value] = _create_node(
            p_data, p_labels, binary_split_threshold=binary_split_threshold, _include=_next_include)

    return BranchNode(feature_idx=feature_idx, branches=nodes, majority_label=_most_common(y))


class MMDecisionTree:
    def __init__(self, binary_split_threshold: int = 10, untrained_value_mode: UntrainedValueMode = "majority"):
        """A decision tree classifier

        Parameters
        ----------
            `untrained_value_mode` : UntrainedValueMode
        """
        self._tree: BaseNode = None
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None
        self._binary_split_threshold = binary_split_threshold
        self._untrained_value_mode = untrained_value_mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_data = X
        self._train_labels = y
        self._tree = _create_node(
            X, y, binary_split_threshold=self._binary_split_threshold
        )

        if DEBUG:
            DEBUG_PRINT("Tree created with", self._tree.count(), "nodes")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            self._tree.predict(X[i, :], self._untrained_value_mode)
            for i in range(X.shape[0])
        ])


class MMRandomForest:
    def __init__(self, n_trees: int = 100, bootstrap_size: float = 0.75, binary_split_threshold: int = 10, untrained_value_mode: UntrainedValueMode = "majority", multiprocessing_limit: int = 8):
        self._forest: list[MMDecisionTree] = []
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None
        self._n_trees = n_trees
        self._bootstrap_size = bootstrap_size
        self._binary_split_threshold = binary_split_threshold
        self._untrained_value_mode = untrained_value_mode
        self._multiprocessing_limit = multiprocessing_limit

    def _create_tree(self, idx: int) -> MMDecisionTree:
        if DEBUG:
            DEBUG_PRINT("Training tree", idx + 1, "of", self._n_trees)

        random = default_rng()

        samples = random.choice(
            self._train_data.shape[0],
            int(self._train_data.shape[0] *
                np.min([1, self._bootstrap_size])),
            replace=False
        )
        tree = MMDecisionTree(binary_split_threshold=self._binary_split_threshold,
                              untrained_value_mode=self._untrained_value_mode)
        tree.fit(self._train_data[samples],
                 self._train_labels[samples])
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEBUG:
            start = datetime.datetime.now()
            DEBUG_PRINT("Training started at", start)

        self._train_data = X
        self._train_labels = y

        # Use multiprocessing to create the trees for quicker training
        with Pool(self._multiprocessing_limit) as pool:
            self._forest = pool.map(self._create_tree, range(self._n_trees))

        if DEBUG:
            DEBUG_PRINT("Training complete in",
                        datetime.datetime.now() - start)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        forest_predictions = np.array(
            [tree.predict(X) for tree in self._forest])
        # Return the most common label for each column
        return np.array([
            _most_common(forest_predictions[:, i]) for i in range(forest_predictions.shape[1])
        ])
