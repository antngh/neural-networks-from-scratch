from collections import defaultdict
from random import shuffle
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output

from neural_networks_from_scratch.models.model_base import GradientModelBase


def train_model(
    model: GradientModelBase,
    x: list,
    y_true: list,
    epochs: int,
    learning_rate: float,
    loss_function: Callable[[float, float], float],
    batch_size: int | None = None,
    learning_rate_decay: float = 1.0,
    record_steps: int | None = None,
    plot: bool = True,
) -> tuple[float, pd.DataFrame]:
    """
    Utility function to train a model.

    Notes:
    - This takes the entire data as one batch, so one epoch is on training step, and we
      are doing gradient descent but not mini-batch/stochastic gradient descent.

    Parameters
    ----------
    model : GradientModelBase
        The model to train.
    x : list
        The input data.
    y_true : list
        The true labels
    epochs : int
        The number of epochs to train for.
    learning_rate : float
        The learning rate.
    loss_function : Callable[[float, float], float]
        The loss function, a function of the form f(y_true, y_pred) -> loss.
    learning_rate_decay : float | None, optional
        The learning rate decay factor, by default 1.0.
    record_steps : int | None, optional
        The number of steps between each record, by default None.
    plot : bool, optional
        Whether to plot the loss, by default True.

    Returns
    -------
    tuple[float, pd.DataFrame]
        The final loss and a dataframe of the loss values.
    """
    res = defaultdict(list)

    record_steps = record_steps or epochs / 100

    batch_size = batch_size or len(x)

    print(f"{batch_size=}")

    model

    # Shuffle the list in place
    combined_data = list(zip(x, y_true))
    for i in range(epochs):
        shuffle(combined_data)
        x, y_true = [list(data_) for data_ in zip(*combined_data)]
        n_batches_per_epoch = max(len(x) // batch_size, 1)
        for minibatch_i in range(n_batches_per_epoch):
            if i * n_batches_per_epoch % (epochs * n_batches_per_epoch // 5) == 0:
                learning_rate *= learning_rate_decay

            slice = [minibatch_i * batch_size, (minibatch_i + 1) * batch_size]
            print(f"{slice=}")
            y_pred = model.forward(x[slice[0] : slice[1]])

            loss = loss_function(y_pred=y_pred, y_true=y_true[slice[0] : slice[1]])
            loss.set_is_network_output()

            if i % record_steps == 0:
                if plot:
                    print(f"{i=}, {learning_rate=}, {loss.val=}")

                res["i"].append(i + slice[0] / (n_batches_per_epoch * batch_size))
                res["loss"].append(loss.val)

                if plot:
                    clear_output(wait=True)
                    plt.clf()

                    res_df = pd.DataFrame(res)
                    ax = sns.lineplot(data=res_df, x="i", y="loss")
                    ax.set_yscale("log")
                    plt.show()

            loss.update_full_network(
                learning_rate, clear_grad=True, clear_downstream_data=True
            )

    loss.set_track_gradients_full_network(False)
    return loss, pd.DataFrame(res)
