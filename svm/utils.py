from typing import Tuple, Union, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import sklearn


def make_plot(
        models: Tuple[Union[sklearn.svm.SVC, sklearn.svm.LinearSVC], ...],
        x: np.ndarray,
        y: np.ndarray,
        poly_degree: int
) -> None:
    """
    Make plot of Iris dataset using different SVM kernels
    :param models: Tuple[Union[sklearn.svm.SVC, sklearn.svm.LinearSVC], ...], tuple of svm classifiers
    :param x: np.ndarray, features
    :param y: np.ndarray, targets
    :param poly_degree: int, degree of polynomial kernel
    :return: None
    """
    titles = (
        'SVC with linear kernel',
        'LinearSVC (linear kernel)',
        'SVC with RBF kernel',
        f'SVC with polynomial (degree {poly_degree}) kernel'
    )

    fig, sub = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    x0, x1 = x[:, 0], x[:, 1]
    xx, yy = _make_meshgrid(x0, x1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        _plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()

    return None


def make_plot_rbf(
        x: np.ndarray,
        y: np.ndarray,
        gamma_list: Sequence[float],
        c_list: Sequence[float]
) -> None:
    """
    Make plots for multiple values of c and gamma with RBF kernel
    :param x: np.ndarray, features
    :param y: np.ndarray, target
    :param gamma_list: Sequence[float], sequence of gamma values to plot
    :param c_list:Sequence[float], sequence of c values to plot
    :return: None
    """
    n = len(gamma_list) * len(c_list) * 0.5
    n_rows = int(np.ceil(n))

    fig, sub = plt.subplots(n_rows, 2, figsize=(15, 100))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    axes = sub.flatten()
    idx = 0

    x0, x1 = x[:, 0], x[:, 1]
    xx, yy = _make_meshgrid(x0, x1)

    for gamma in gamma_list:
        for c in c_list:
            ax = axes[idx]

            clf = sklearn.svm.SVC(kernel='rbf', gamma=gamma, C=c)
            clf.fit(x, y)

            _plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('Sepal length')
            ax.set_ylabel('Sepal width')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(f'SVC with RBF kernel: gamma={gamma}, C={c}')
            idx += 1

    plt.show()

    return None


def _make_meshgrid(
        x: np.ndarray,
        y: np.ndarray,
        step: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh of points to plot in
    :param x: np.ndarray, first array to create meshgrid from
    :param y: np.ndarray, second array to create meshgrid from
    :param step: float, step size in meshgrid
    :return: Tuple[np.ndarray, np.ndarray], tuple of meshgrids
    """
    x_min, x_max = min(x) - 1, max(x) + 1
    y_min, y_max = min(y) - 1, max(y) + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    return xx, yy


def _plot_contours(
        ax: plt.Axes,
        clf: Union[sklearn.svm.SVC, sklearn.svm.LinearSVC],
        xx: np.ndarray,
        yy: np.ndarray,
        **kwargs: Optional
) -> None:
    """
    Plot the decision boundaries for a classifier.
    :param ax: plt.Axes, axis to plot onto
    :param clf: Union[sklearn.svm.SVC, sklearn.svm.LinearSVC], sklearn classifiers
    :param xx: np.ndarray, x values to plot onto
    :param yy: np.ndarray, y values to plot onto
    :param kwargs: Optional, kwargs
    :return: None
    """
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, **kwargs)

    return None
