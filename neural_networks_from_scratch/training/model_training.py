from collections import defaultdict
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
    train_steps: int,
    learning_rate: float,
    loss_function: Callable[[float, float], float],
    learning_rate_decay: float = 1.0,
    record_steps: bool | None = None,
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
    train_steps : int
        The number of training steps.
    learning_rate : float
        The learning rate.
    loss_function : Callable[[float, float], float]
        The loss function, a function of the form f(y_true, y_pred) -> loss.
    learning_rate_decay : float | None, optional
        The learning rate decay factor, by default 1.0.
    record_steps : bool | None, optional
        The number of steps between each record, by default None.
    plot : bool, optional
        Whether to plot the loss, by default True.

    Returns
    -------
    tuple[float, pd.DataFrame]
        The final loss and a dataframe of the loss values.
    """
    res = defaultdict(list)

    record_steps = record_steps or train_steps / 100

    for i in range(train_steps):
        if i % (train_steps // 5) == 0:
            learning_rate *= learning_rate_decay

        y_pred = model.forward(x)
        loss = loss_function(y_pred=y_pred, y_true=y_true)
        loss.set_is_network_output()

        if i % record_steps == 0:
            if plot:
                print(f"{i=}, {learning_rate=}, {loss.val=}")

            res["i"].append(i)
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
