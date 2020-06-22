import shap
import pandas as pd
import scipy
import numpy as np
import matplotlib.pylab as pl

from shap.plots import labels, colors
from shap.plots.monitoring import truncate_text


def subsample_data(X, y, n_sample=100, seed_temp=1234):
    """Subsample data, stratified by target variable y"""
    frac = n_sample / X.shape[0]

    data = X.copy(deep=True)
    data["label"] = y
    strata = ["label"]

    data_subsample = data.groupby(strata, group_keys=False)\
        .apply(lambda x: x.sample(frac=frac, replace=False, random_state=seed_temp))

    # assert
    #print("imbalance ratio()")

    return data_subsample[X.columns]


def plot_shap_loss(feat_name, shap_values_logloss, data_X):
    """make shap monitoring plot for features"""
    ind = data_X.columns.get_loc(feat_name)
    shap.monitoring_plot(ind=ind, shap_values=shap_values_logloss, features=data_X)


def plot_shap_loss_compare(feat_name, list_ShapValue_Features,
                           use_fine_tune=False, inc_input=50):
    """make shap monitoring plots"
    
    Parameters
    ----------
    feat_name: str or list
        features
    list_ShapValue_Features: list
        a list of list. Each sub-list consists of two Pandas Dataframe,
        [[shap_values_logloss_1, data_X_1], [shap_values_logloss_2, data_X_2], ...]
    use_fine_tune: bool
        whether to use the customized monitoring plot, with a customized increment
    inc_input: int
        customized increment of data points instead of the default 50

    Returns
    -------

    """""

    index_boundary = []
    i_snapshot = 0

    for snapshot in list_ShapValue_Features:
        shap_values_logloss_snapshot = snapshot[0]
        data_X_snapshot = snapshot[1]

        if i_snapshot == 0:
            shap_values_logloss_total = shap_values_logloss_snapshot.copy(deep=True)
            data_X_total = data_X_snapshot.copy(deep=True)
        else:
            # mark the index
            index_boundary.append(shap_values_logloss_total.shape[0] - 1)

            shap_values_logloss_total = pd.concat([shap_values_logloss_total,
                                                   shap_values_logloss_snapshot],
                                                  axis=0, ignore_index=True)

            data_X_total = pd.concat([data_X_total, data_X_snapshot],
                                     axis=0, ignore_index=True)

        i_snapshot = i_snapshot + 1

    if isinstance(feat_name, str):
        feat_name = [feat_name]

    if use_fine_tune:
        print("now change the increment of data points for the t- test as: ", inc_input,
              "default by shap package is 50")

    for each_feat in feat_name:
        ind = data_X_total.columns.get_loc(each_feat)

        if use_fine_tune:
            monitoring_plot_tuned(ind=ind, shap_values=shap_values_logloss_total.values, features=data_X_total,
                                  inc_input=inc_input)
        else:
            shap.monitoring_plot(ind=ind, shap_values=shap_values_logloss_total.values, features=data_X_total)

    # TODO: plot the vertical line to indicate the time boundary
    print("snapshot boundary (index): ", index_boundary)


def monitoring_plot_tuned(ind, shap_values, features, feature_names=None, show=True, inc_input=50):
    """ Create a SHAP monitoring plot, modified by lcx.

    (Note this function is preliminary and subject to change!!)
    A SHAP monitoring plot is meant to display the behavior of a model
    over time. Often the shap_values given to this plot explain the loss
    of a model, so changes in a feature's impact on the model's loss over
    time can help in monitoring the model's performance.
    Parameters
    ----------
    ind : int
        Index of the feature to plot.
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)
    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)
    feature_names : list
        Names of the features (length # features)
    inc_input: int
        Increment of data points
    """


    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values

    num_features = shap_values.shape[1]

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    pl.figure(figsize=(10, 3))
    ys = shap_values[:, ind]
    xs = np.arange(len(ys))  # np.linspace(0, 12*2, len(ys))

    pvals = []
    # lcx: change the increment of data points
    # inc = 50
    inc = inc_input

    for i in range(inc, len(ys) - inc, inc):
        # stat, pval = scipy.stats.mannwhitneyu(v[:i], v[i:], alternative="two-sided")
        stat, pval = scipy.stats.ttest_ind(ys[:i], ys[i:])
        pvals.append(pval)
    min_pval = np.min(pvals)
    min_pval_ind = np.argmin(pvals) * inc + inc

    if min_pval < 0.05 / shap_values.shape[1]:
        pl.axvline(min_pval_ind, linestyle="dashed", color="#666666", alpha=0.2)
        # lcx: print the min_pval and min_pval_ind for checking
        print("t-test fail:")
        print("min_pval: ", min_pval, ";", "min_pval_ind: ", min_pval_ind)

    pl.scatter(xs, ys, s=10, c=features[:, ind], cmap=colors.red_blue)

    pl.xlabel("Sample index")
    pl.ylabel(truncate_text(feature_names[ind], 30) + "\nSHAP value", size=13)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    cb = pl.colorbar()
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 20)
    cb.set_label(truncate_text(feature_names[ind], 30), size=13)
    if show:
        pl.show()