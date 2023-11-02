import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from typing import List, Tuple

warnings.simplefilter("ignore")


def plot_batch(
    images: tf.Tensor | np.ndarray,
    class_names: list = None,
    labels: tf.Tensor | np.ndarray = None,
    ncols: int = 1,
    figsize: Tuple[int, int] = (10, 10),
    names: List[str] = None,
    float_: bool = False,
    **kw,
) -> None:
    assert (
        labels is not None and class_names is not None
    ) or names is not None, "Either labels or names must be provided"

    n = images.shape[0]
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=figsize)

    if isinstance(images, tf.Tensor):
        images = images.numpy()

    if not float_:
        images = images.astype("uint8")

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i], **kw)
        plt.title(class_names[labels[i]] if names is None else names[i])
        plt.axis("off")


def plot(
    arrs: List[np.array], names: list = None, figsize: Tuple[int, int] = (3, 3), **kw
) -> None:
    plt.figure(figsize=figsize)
    arrs = [arrs] if not isinstance(arrs, list) else arrs
    if names is None:
        names = ["" for _ in range(len(arrs))]

    for arr, name in zip(arrs, names):
        if name != "":
            plt.plot(arr, label=name, **kw)
        else:
            plt.plot(arr, **kw)

    if names[0] != "":
        plt.legend(loc="upper right", bbox_to_anchor=(1.6, 1))
    plt.show()


def implot(
    imgs: List[np.array],
    names: list = None,
    cols: int = None,
    figsize: Tuple[int, int] = None,
    show_val: bool = False,
    font_size: int = 4,
    **kw,
) -> None:
    imgs = [imgs] if not isinstance(imgs, list) else imgs
    cols = 1 if cols is None else cols
    rows = int(np.ceil(len(imgs) / cols))

    figsize = (3 * cols, 3 * rows) if figsize is None else figsize
    plt.figure(figsize=figsize)

    if names is None:
        names = ["" for _ in range(len(imgs))]

    for i, (arr, name) in enumerate(zip(imgs, names)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(arr, **kw)
        plt.axis("off")
        plt.title(name)

        if show_val:
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    plt.text(
                        x,
                        y,
                        f"{arr[y, x]:1.1f}",
                        color="red",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                    )
    plt.show()


def apply(
    imgs: list,
    kernel_x: np.array = None,
    kernel_y: np.array = None,
    filter_x: callable = None,
    filter_y: callable = None,
    filter_all: callable = None,
    **kw,
) -> None:
    assert (kernel_x is not None and kernel_y is not None) or (
        filter_x is not None and filter_y is not None and filter_all is not None
    ), "You must provide either a kernel or a filter"

    # vertical gradient
    apply_x = lambda img: (
        cv2.filter2D(img, -1, kernel_x) if kernel_x is not None else filter_x(img)
    )
    # horizontal gradient
    apply_y = lambda img: (
        cv2.filter2D(img, -1, kernel_y) if kernel_y is not None else filter_y(img)
    )

    # magnitude - how quickly pixel changes at given point
    apply_all = lambda img: (
        np.sqrt(apply_x(img) ** 2 + apply_y(img) ** 2)
        if filter_all is None
        else filter_all(img)
    )

    applied_imgs = [
        apply_(img) for apply_ in (apply_x, apply_y, apply_all) for img in imgs
    ]

    implot(
        applied_imgs,
        names=["vertical gradient"] * len(imgs)
        + ["horizontal gradient"] * len(imgs)
        + ["magnitude"] * len(imgs),
        cols=len(imgs),
        **kw,
    )


def im_hist_plot(imgs: list, names: list = None, figsize=None, **kw) -> None:
    cols = 2
    rows = len(imgs)

    figsize = (3 * cols, 3 * rows) if figsize is None else figsize
    plt.figure(figsize=figsize)

    if names is None:
        names = ["" for _ in range(len(imgs))]

    for i, (img, title) in enumerate(zip(imgs, names)):
        plt.subplot(rows, cols, 1 + 2 * i)
        plt.title(title)
        plt.axis("off")
        plt.imshow(img, cmap="gray")
        plt.subplot(rows, cols, 2 + 2 * i)
        hist = np.histogram(img, bins=np.arange(0, 256))
        plt.plot(hist[1][:-1], hist[0], lw=2)
    plt.show()
