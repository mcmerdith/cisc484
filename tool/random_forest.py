from dataclasses import dataclass
import datetime
from functools import reduce
import numpy as np
from sklearn.metrics import accuracy_score
from numpy.random import default_rng

DEBUG = True
VERBOSE = False


def DEBUG_PRINT(*args, verbose: bool = False,  **kwargs):
    if VERBOSE:
        print("[VERBOSE]", *args, **kwargs)
    else:
        print("[DEBUG]", *args, **kwargs)


def DEBUG_SPACER(shape) -> np.ndarray:
    spacer = np.zeros(shape, dtype=np.str_)
    spacer.fill("=")
    return spacer


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
        DEBUG_PRINT(f"Information gain of\n{
            np.c_[x, DEBUG_SPACER(y.shape), y]}\n{information_gain}\n", verbose=True)
    return information_gain


def _partition(x: np.ndarray, y: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Partition `x` according to the feature of `x` with the highest information gain

    Parameters
    ----------
        `x` : np.ndarray<N,M>
            A 2D array containing `N` samples of `M` features

        `y` : np.ndarray<N>
            A 1D array of the label values

    Returns
    -------
        `partitions` : dict[int, tuple[np.ndarray<N,M>, np.ndarray<N>]]
            A dict of `S` partitions of `x` according to the feature of `x`
            with the highest information gain and the corresponding label values
    """

    # Select the highest information gain feature
    selected_feature = np.argmax(_information_gain(x, y))

    if VERBOSE:
        DEBUG_PRINT("Creating partition on feature",
                    selected_feature, "\n", verbose=True)

    # Get the values of the selected feature
    feature = x[:, selected_feature]
    if VERBOSE:
        DEBUG_PRINT(f"Feature values\n{
            np.c_[feature, DEBUG_SPACER(y.shape), y]}\n", verbose=True)

    # Copy the array without the selected feature
    new_data = np.delete(x, selected_feature, axis=1)
    # Partition the new array according to the value of the feature
    partitions = {
        feature_value: (
            new_data[x[:, selected_feature] == feature_value, :],
            y[x[:, selected_feature] == feature_value]
        ) for feature_value in np.unique(feature)
    }

    if VERBOSE:
        DEBUG_PRINT(*[f"\nPartition created for {feature_value}\n{np.c_[data, DEBUG_SPACER(labels.shape), labels]}\n"
                      for feature_value, (data, labels) in partitions.items()], verbose=True)
    return partitions


class BaseNode:
    def _get_branch(self, X: int) -> "BaseNode":
        raise NotImplementedError()

    def predict(self, X: int) -> bool:
        predictor = self._get_branch(X)
        if isinstance(predictor, BaseNode):
            return predictor.predict(X)
        else:
            print("[PANIC] Failed to predict value for", X)
        return predictor


@dataclass
class LeafNode(BaseNode):
    value: bool

    def predict(self, _: int) -> bool:
        return self.value


@dataclass
class StandardNode(BaseNode):
    branches: dict[int, BaseNode]

    def _get_branch(self, X: int) -> BaseNode:
        return self.branches.get(X)


@dataclass
class RealValueNode(BaseNode):
    threshold: int
    left: BaseNode
    right: BaseNode

    def _get_branch(self, X: int) -> BaseNode:
        if X < self.threshold:
            return self.left
        return self.right


def _most_common(x: np.ndarray) -> int:
    possible = np.unique(x)
    frequency = [len(x[x == value]) for value in possible]
    return possible[np.argmax(frequency)]


def _create_node(x: np.ndarray, y: np.ndarray) -> BaseNode:
    # If there is only one label, return a leaf node
    unique_labels = np.unique(y)
    if unique_labels.size == 1:
        if VERBOSE:
            DEBUG_PRINT("only one label", unique_labels, verbose=True)
        return LeafNode(value=unique_labels[0])

    # If there is only one set of inputs, return a leaf node
    unique_values = np.unique(x, axis=0)
    if len(unique_values) == 1:
        if VERBOSE:
            DEBUG_PRINT("only one value", unique_values, verbose=True)
        return LeafNode(value=_most_common(y))

    # Otherwise partition the data
    partitions = _partition(x, y)
    nodes: dict[int, BaseNode] = {}

    if VERBOSE:
        for feature, (data, labels) in partitions.items():
            DEBUG_PRINT(
                f"Partition {feature}\n{np.c_[data, DEBUG_SPACER(labels.shape), labels]}\n", verbose=True)

    # Create the nodes for each partition
    for feature, (data, labels) in partitions.items():
        nodes[feature] = _create_node(data, labels)

    return StandardNode(branches=nodes)


class MMDecisionTree:
    def __init__(self):
        self._tree: BaseNode = None
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_data = X
        self._train_labels = y
        self.tree = _create_node(X, y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class MMRandomForest:
    def __init__(self, n_trees: int = 100, max_per_bag: int = 10):
        self._n_trees = n_trees
        self._max_per_bag = max_per_bag
        self._forest: list[MMDecisionTree] = []
        self._train_data: np.ndarray = None
        self._train_labels: np.ndarray = None
        self._feature_importances: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEBUG:
            start = datetime.datetime.now()
            DEBUG_PRINT("Training started at", start)
        self._train_data = X
        self._train_labels = y
        for _ in range(self._n_trees):
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
