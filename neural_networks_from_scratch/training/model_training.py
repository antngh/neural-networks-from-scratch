from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output


def train_model(
    model,
    x,
    y_true,
    train_steps,
    learning_rate,
    loss_function,
    learning_rate_decay=1.0,
    record_steps=None,
    plot=True,
):
    res = defaultdict(list)

    record_steps = record_steps or train_steps / 100

    for i in range(train_steps):
        if i % (train_steps // 5) == 0:
            learning_rate *= learning_rate_decay

        y_pred = model.forward(x)
        loss = loss_function(y_pred=y_pred, y_true=y_true)
        loss.set_is_graph_output()

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
