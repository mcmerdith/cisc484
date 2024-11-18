from dataclasses import dataclass
import datetime
from functools import reduce
from typing import Literal
import numpy as np
from sklearn.metrics import accuracy_score
from numpy.random import default_rng

DEBUG = True
VERBOSE = False

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


def VERBOSE_PRINT(*args, **kwargs):
    DEBUG_PRINT("[VERBOSE]", *args, **kwargs)


def DEBUG_SPACER(shape) -> np.ndarray:
    spacer = np.zeros(shape, dtype=np.str_)
    spacer.fill("=")
    return spacer


@dataclass
class BaseNode:
    feature_idx: int
    """The index of the feature this node predicts"""

    def _get_branch(self, X: float, untrained_value_mode: UntrainedValueMode = None) -> "BaseNode":
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

        if isinstance(predictor, BaseNode):
            return predictor.predict(sample, untrained_value_mode)
        else:
            print("[PANIC] Failed to predict value of feature",
                  self.feature_idx, sample[self.feature_idx])

        return predictor


@dataclass
class LeafNode(BaseNode):
    """A leaf node in a decision tree

    The value of `feature_idx` is ignored"""
    label: float

    def predict(self, _: np.ndarray, __: UntrainedValueMode = None) -> float:
        return self.label


@dataclass
class StandardNode(BaseNode):
    branches: dict[float, BaseNode]
    majority_label: float

    def _get_branch(self, X: float, untrained_value_mode: UntrainedValueMode = None) -> BaseNode:
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
class RealValueNode(BaseNode):
    threshold: float
    left: BaseNode
    right: BaseNode

    def _get_branch(self, X: float, _: UntrainedValueMode = None) -> BaseNode:
        if X < self.threshold:
            return self.left
        return self.right


def _n_log_n(n: float) -> float:
    """Return the value of `n*log_2(n)`"""
    return n * np.log2(n)


def _entropy(n: np.ndarray) -> float:
    """Calculate the entropy of `n`

    Parameters
    ----------
        `n` : np.ndarray<1>
            A 1D array of values
    """

    return -1 * sum([
        _n_log_n(len(n[n == label]) / n.size)
        for label in np.unique(n)
    ])


def _conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the entropy of `y` given `x`

    Parameters
    ----------
        `x` : np.ndarray<1>
            A 1D array of the feature values

        `y` : np.ndarray<1>
            A 1D array of the label values

    Restrictions
    ----------
        `x` and `y` must have the same size
    """
    assert x.size == y.size, "x and y must have the same size"

    conditional_probability = []
    for x_i in np.unique(x):
        y_x = y[x == x_i]
        conditional_probability.append(((x[x == x_i]).size / x.size) * sum([
            _n_log_n((y_x[y_x == y_x_i]).size / y_x.size)
            for y_x_i in np.unique(y_x)
        ]))

    return -1 * sum(conditional_probability)


def _information_gain(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Calculate the information gain of `x` given `y`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<1>
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

    # Calculate entropy of x given y
    information_gain = [
        entropy_y - _conditional_entropy(x[:, i], y)
        for i in range(x.shape[1])
    ]
    if VERBOSE:
        VERBOSE_PRINT(
            f"Information gain of\n{np.c_[x, DEBUG_SPACER(y.shape), y]}\n{
                information_gain}\n"
        )
    return information_gain


def _partition(x: np.ndarray, y: np.ndarray, feature: int) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Partition `x` according to `feature`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values 

        `feature` : int
            The index of the feature to partition on

    Returns
    -------
        `partitions` : dict[float, tuple[np.ndarray<N,M>, np.ndarray<N>]]
            A dict of `S` partitions of `x` according to the feature of `x`
            with the highest information gain and the corresponding label values
    """

    if VERBOSE:
        VERBOSE_PRINT("Creating partition on feature", feature, "\n")

    # Get the values of the selected feature
    feature_set = x[:, feature]
    if DEBUG:
        DEBUG_PRINT("Feature value shape", feature_set.shape)
    if VERBOSE:
        VERBOSE_PRINT(
            f"Feature values\n{np.c_[feature_set, DEBUG_SPACER(y.shape), y]}\n"
        )

    # Partition the new array according to the value of the feature
    partitions = {
        feature_value: (
            x[feature_set == feature_value, :],
            y[feature_set == feature_value]
        ) for feature_value in np.unique(feature_set)
    }

    if VERBOSE:
        VERBOSE_PRINT(*[f"\nPartition created for {feature_value}\n{np.c_[data, DEBUG_SPACER(labels.shape), labels]}\n"
                      for feature_value, (data, labels) in partitions.items()])
    return partitions


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


def _create_node(x: np.ndarray, y: np.ndarray, *, _include: np.ndarray = None) -> BaseNode:
    """Create a decision tree node from `x` and `y`

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<1>
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

    ### Base Cases ###

    # If there is only one label, return a leaf node
    unique_labels = np.unique(y)
    if unique_labels.size == 1:
        if VERBOSE:
            VERBOSE_PRINT("only one label", unique_labels)
        return LeafNode(feature_idx=None, label=unique_labels[0])

    # If there is only one set of inputs, return a leaf node
    unique_values = np.unique(x, axis=0)
    if len(unique_values) == 1:
        if VERBOSE:
            VERBOSE_PRINT("only one value", unique_values)
        return LeafNode(feature_idx=None, label=_most_common(y))

    ### Recursive Behavior ###

    # Otherwise partition the data
    # Select the highest information gain feature

    if _include is None:
        if DEBUG:
            DEBUG_PRINT("Initializing to include all features")
        # initial state is to include all features
        _include = np.array([True] * x.shape[1])
        information_gain = _information_gain(x, y)
        if DEBUG:
            DEBUG_PRINT("Information gain", information_gain)
        feature_idx = np.argmax(information_gain)
    else:
        if DEBUG:
            DEBUG_PRINT("Using features", _include)
        # get the feature index with the highest information gain
        feature_idx = np.argmax(_information_gain(x[:, _include], y))
        # convert the feature index to the original index
        feature_idx = _include.nonzero()[0][feature_idx]

    # exclude the feature from the next iteration
    _include[feature_idx] = False

    if DEBUG:
        DEBUG_PRINT("Branching on feature", feature_idx)

    partitions = _partition(x, y, feature_idx)
    nodes: dict[float, BaseNode] = {}

    if DEBUG:
        DEBUG_PRINT("Creating", len(partitions), "partitions")
        for feature_value, (p_data, p_labels) in partitions.items():
            DEBUG_PRINT(f"Partition {feature_value} contains {
                        p_data.shape[0]} samples and {p_labels.shape[0]} labels")
            if VERBOSE:
                VERBOSE_PRINT(
                    f"Partition {feature_value}\n{
                        np.c_[p_data, DEBUG_SPACER(p_labels.shape), p_labels]}\n"
                )

    # Create the nodes for each partition
    for feature_value, (p_data, p_labels) in partitions.items():
        nodes[feature_value] = _create_node(
            p_data, p_labels, _include=_include)

    return StandardNode(feature_idx=feature_idx, branches=nodes, majority_label=_most_common(y))


class MMDecisionTree:
    def __init__(self, untrained_value_mode: UntrainedValueMode = "nearest"):
        """A decision tree classifier

        Parameters
        ----------
            `untrained_value_mode` : "majority" | "nearest"
                How the model should handle unknown values
                "majority" will return the most common label
                "nearest" will return the label closest to the unknown value
        """
        self._tree: BaseNode = None
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None
        self._untrained_value_mode = untrained_value_mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_data = X
        self._train_labels = y
        self.tree = _create_node(X, y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            self.tree.predict(X[i, :], self._untrained_value_mode)
            for i in range(X.shape[0])
        ])


class MMRandomForest:
    def __init__(self, n_trees: int = 100, max_per_bag: int = 10):
        self._n_trees = n_trees
        self._max_per_bag = max_per_bag
        self._forest: list[MMDecisionTree] = []
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEBUG:
            start = datetime.datetime.now()
            DEBUG_PRINT("Training started at", start)
        self._train_data = X
        self._train_labels = y
        for i in range(self._n_trees):
            if DEBUG:
                DEBUG_PRINT("Training tree", i)
            random = default_rng()
            samples = random.choice(
                self._train_data.shape[0], self._max_per_bag, replace=False)
            tree = MMDecisionTree()
            tree.fit(self._train_data[samples], self._train_labels[samples])
            self._forest.append(tree)

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


if __name__ == "__main__":
    data = np.array([[1, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([1, 1, 1, 1, 1, 0])

    tree = MMRandomForest()
    tree.fit(data, labels)
    print(tree.predict(data))
