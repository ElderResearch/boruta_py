#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

from __future__ import print_function, division
import numpy as np
import scipy as sp
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble.forest import _get_n_samples_bootstrap, _generate_unsampled_indices
from sklearn.metrics import accuracy_score, mean_squared_error
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from distutils.version import LooseVersion


class BorutaPy(BaseEstimator, TransformerMixin):
    """
    Improved Python implementation of the Boruta R package.

    The improvements of this implementation include:
    - Faster run times:
        Thanks to scikit-learn's fast implementation of the ensemble methods.
    - Scikit-learn like interface:
        Use BorutaPy just like any other scikit learner: fit, fit_transform and
        transform are all implemented in a similar fashion.
    - Modularity:
        Any ensemble method could be used: random forest, extra trees
        classifier, even gradient boosted trees.
    - Two step correction:
        The original Boruta code corrects for multiple testing in an overly
        conservative way. In this implementation, the Benjamini Hochberg FDR is
        used to correct in each iteration across active features. This means
        only those features are included in the correction which are still in
        the selection process. Following this, each that passed goes through a
        regular Bonferroni correction to check for the repeated testing over
        the iterations.
    - Percentile:
        Instead of using the max values of the shadow features the user can
        specify which percentile to use. This gives a finer control over this
        crucial parameter. For more info, please read about the perc parameter.
    - Automatic tree number:
        Setting the n_estimator to 'auto' will calculate the number of trees
        in each itartion based on the number of features under investigation.
        This way more trees are used when the training data has many feautres
        and less when most of the features have been rejected.
    - Ranking of features:
        After fitting BorutaPy it provides the user with ranking of features.
        Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
        starting from 3, based on their feautre importance history through
        the iterations.

    We highly recommend using pruned trees with a depth between 3-7.

    For more, see the docs of these functions, and the examples below.

    Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error.

    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (yes, minimal optimal set of features
    by definition depends on your classifier choice).

    Parameters
    ----------

    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        feature_importances_ attribute. Important features must correspond to
        high absolute values in the feature_importances_.

    n_estimators : int or string, default = 100
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.

    perc : int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.

    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.

    two_step : Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.

    max_iter : int, default = 100
        The number of maximum iterations to perform.

    random_state : int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already
    
    importance_type : str, default = 'auto'
        Should be one of auto, permutation_oob, permutation_holdout, permutation_bytree.
        auto uses the default estimator feature_importances_ attribute;
        permutation_oob uses permutation importance based on oob_accuracy as 
        implemented in the rfpimp module;
        permutation_holdout uses permutation importance based on holdout accuracy
        by splitting the dataset in two;
        permutation_bytree uses permutation importance for each tree individually
        and averages the results. I believe this is what is implemented in the 
        R ranger package according to: 
        <https://github.com/imbs-hl/ranger/issues/237#issuecomment-344717299>
    
    scale_permutation_bytree : bool, default = False
        Whether or not to scale permutation importance by standard error.
        This is only relevant when importance_type = permutation_bytree.
        The R Boruta function scales, but by default it's set to False
        in ranger. The literature indicates scaling biases the results.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]

        The mask of selected features - only confirmed ones are True.

    support_weak_ : array of shape [n_features]

        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations..

    ranking_ : array of shape [n_features]

        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and tentative features are assigned
        rank 2.

    Examples
    --------

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check selected features - first 5 features are selected
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=100, perc=100, alpha=0.05, two_step=True, 
    max_iter=100, random_state=None, verbose=0, importance_type='gini', scale_permutation_bytree=False):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.importance_type = importance_type
        self.scale_permutation_bytree = scale_permutation_bytree
        if is_classifier(self.estimator):
            self.task = 'classification'
        elif is_regressor(self.estimator):
            self.task = 'regression'
        else:
            self.task = 'other'

    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """

        return self._fit(X, y)

    def transform(self, X, weak=False):
        """
        Reduces the input X to the features selected by Boruta.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak)

    def fit_transform(self, X, y, weak=False):
        """
        Fits Boruta, then reduces the input X to the selected features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, weak)

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)
        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        # sha_history records min, mean, max like the R implementation
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_history = np.zeros(3, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            self.estimator.set_params(random_state=self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp, cur_sha = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_sha, self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp))
            sha_history = np.vstack((
                sha_history, 
                [cur_sha.min(), cur_sha.mean(), cur_sha.max()]))
            
            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(
            tentative_median > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1
        # calculate ranks in each iteration, then median of ranks across feats
        if imp_history_rejected.size == 0:
            ranks = np.ones(n_feat)
        else:
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks

        # save imp history for plotting purposes
        self._imp_history = imp_history
        self._sha_history = sha_history

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self

    def _transform(self, X, weak=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            X = X[:, self.support_ + self.support_weak_]
        else:
            X = X[:, self.support_]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth == None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def permutation_importances_bytree(self, X_train, y_train):
        """Permutation Importance by tree

        For a given feature, averages the loss in accuracy when 
        permuting that feature on each individual tree. Same 
        as ranger importance = 'permutation', as far as I can
        tell.

        Parameters
        ----------
        X_train : np.array
            [description]
        y_train : np.array
            [description]

        Returns
        -------
        imp : np.array
            array of feature importances
        """
        imp = []
        n_samples = len(X_train)
        n_classes = len(np.unique(y_train))
        for col in range(-1, X_train.shape[1]):
            metrics = []
            if col >= 0:
                save = X_train[:, col].copy()
            for i, tree in enumerate(self.estimator.estimators_):
                if col >= 0:
                    X_train[:, col] = np.random.permutation(X_train[:, col])
                unsampled_indices = self._get_unsampled_indices(tree, n_samples)
                tree_preds = tree.predict(X_train[unsampled_indices, :])
                if self.task == 'classification':
                    curr_metric = (y_train[unsampled_indices] == tree_preds).mean()
                elif self.task == 'regression':
                    curr_metric = -1 * ((y_train[unsampled_indices] - tree_preds) ** 2).mean()
                else:
                    raise ValueError(f'Permutation importance not supported for {self.estimator}')
                if col >= 0:
                    metrics.append(baseline_metrics[i] - curr_metric)
                else:
                    metrics.append(curr_metric)
            if self.scale_permutation_bytree:
                m = np.mean(metrics) / np.std(metrics, ddof=1)
            else:
                m = np.mean(metrics)
            if col >= 0:
                X_train[:, col] = save
                imp.append(m)
            else:
                baseline_metrics = list(metrics)

        return np.array(imp)

    def permutation_importances_oob(self, X_train, y_train):
        """ Permutation Importance using OOB data

        Taken from rfpimp module to decouple dependency and modify
        <https://github.com/parrt/random-forest-importances>
        ---------------------------------------------------------------
        For each feature, calculates difference between baseline
        oob accuracy and oob accuracy when that feature is permuted.        

        Parameters
        ----------
        X_train : np.array
            [description]
        y_train : np.array
            [description]

        Returns
        -------
        imp : np.array
            array of feature importances
        """
        if self.task == 'classification':
            oob_metric = self.oob_classifier
        elif self.task == 'regression':
            oob_metric = self.oob_regressor
        else:
            raise ValueError(f'Permutation importance not supported for {self.estimator}')
        baseline = oob_metric(X_train, y_train)
        imp = []
        for col in range(X_train.shape[1]):
            save = X_train[:, col].copy()
            X_train[:, col] = np.random.permutation(X_train[:, col])
            m = oob_metric(X_train, y_train)
            X_train[:, col] = save
            drop_in_metric = baseline - m
            imp.append(drop_in_metric)
        return np.array(imp)


    def _get_unsampled_indices(self, tree, n_samples):
        """
        Taken from rfpimp module to decouple dependency and modify
        <https://github.com/parrt/random-forest-importances>
        ---------------------------------------------------------------
        An interface to get unsampled indices regardless of sklearn version.
        """
        if LooseVersion(sklearn.__version__) >= LooseVersion("0.22"):
            # Version 0.22 or newer uses 3 arguments.
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
            return _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)
        else:
            # Version 0.21 or older uses only two arguments.
            return _generate_unsampled_indices(tree.random_state, n_samples)


    def oob_classifier(self, X, y):
        """
        Taken from rfpimp module to decouple dependency and modify
        <https://github.com/parrt/random-forest-importances>
        ---------------------------------------------------------------
        Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
        classifier. We learned the guts of scikit's RF from the BSD licensed
        code:
        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
        """
        n_samples = len(X)
        n_classes = len(np.unique(y))
        predictions = np.zeros((n_samples, n_classes))
        for tree in self.estimator.estimators_:
            unsampled_indices = self._get_unsampled_indices(tree, n_samples)
            tree_preds = tree.predict_proba(X[unsampled_indices, :])
            predictions[unsampled_indices] += tree_preds

        predicted_class_indexes = np.argmax(predictions, axis=1)
        predicted_classes = [self.estimator.classes_[i] for i in predicted_class_indexes]

        oob_score = np.mean(y == predicted_classes)
        return oob_score

    def oob_regressor(self, X, y):
        """
        Compute out-of-bag (OOB) MSE for a scikit-learn random forest
        regressor. We learned the guts of scikit's RF from the BSD licensed
        code:
        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
        """
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)
        for tree in self.estimator.estimators_:
            unsampled_indices = self._get_unsampled_indices(tree, n_samples)
            tree_preds = tree.predict(X[unsampled_indices, :])
            predictions[unsampled_indices] += tree_preds
            n_predictions[unsampled_indices] += 1

        predictions /= n_predictions
        oob_score = -1 * ((y - predictions) ** 2).mean()
        return oob_score


    def _get_imp(self, X, y):
        if self.importance_type == 'permutation_holdout':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        else:
            X_train, y_train = X.copy(), y.copy()
        try:
            self.estimator.fit(X_train, y_train)
        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided'
                             'estimator cannot be fitted to your data.\n' + str(e))
        if self.importance_type == 'gini':
            try:
                imp = self.estimator.feature_importances_
            except Exception:
                raise ValueError('Only methods with feature_importance_ attribute '
                                'are currently supported in BorutaPy.')
        elif self.importance_type == 'permutation_oob':
            imp = self.permutation_importances_oob(X, y)
        elif self.importance_type == 'permutation_holdout':
            imp = permutation_importance(self.estimator, X_test, y_test, n_repeats=5, n_jobs=-1)['importances_mean']
        elif self.importance_type == 'permutation_bytree':
            imp = self.permutation_importances_bytree(X, y)
        else:
            raise ValueError('importance_type should be "gini", "permutation_oob", "permutation_holdout", or "permutation_bytree"')
        return imp

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, dec_reg):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        imp = self._get_imp(np.hstack((x_cur, x_sha)), y)
        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]
        return imp_real, imp_sha

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.

        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate

        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\nBorutaPy finished running.\n\n" + result
        print(output)

    def plot_importances(self, feature_names=None):
        """Produce a plot of importances, including shadows.
        
        Params:
            feature_names  iterable of feature names or None. If None, 
                autogenerated names are used.
        
        Returns:
            a matplotlib figure
        
        """
        imps = self.get_historical_importances(feature_names)
        dims = (9, 6)
        fig, ax = plt.subplots(figsize=dims)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Importance (Relevance)', fontsize=14)
        plt.title("Boruta Feature Importances", fontsize=16)
        plt.xticks(fontsize=12)
        fig = sns.boxplot(data=imps, orient='v', ax=ax, palette='viridis')
        fig.set_xticklabels(fig.get_xticklabels(), rotation='vertical')
        sns.despine()
        return fig

    def get_historical_importances(self, feature_names=None):
        """Produce a pandas DataFrame containing the importance history.

        This method binds in shadow-variable statistics as well.
        
        Params:
            feature_names  iterable of feature names or None. If None, 
                autogenerated names are used.
        
        Returns:
            a pandas data frame of importance history
        
        """
        if feature_names is None:
            feature_names = self._automatic_imp_history_names()
        else:
            feature_names = list(feature_names)
        
        # Guarantee the correct number of labels
        if len(feature_names) != self._imp_history.shape[1]:
            raise ValueError('number of feature names should match ' +
                'the number of columns in the importance history')
        
        # Bind feature and shadow importances together, respectively
        feature_names += self._shadow_names
        imps = np.hstack((self._imp_history, self._sha_history))

        imps = pd.DataFrame(imps, columns=feature_names)
        imps_idx = imps.mean().sort_values(ascending=False).index
        imps = imps.reindex(columns=imps_idx)
        
        # first row is all zeroes
        imps = imps.iloc[1:]
        return imps
    
    def _automatic_imp_history_names(self):
        """
        Produce autogenerated variable names for padas and matplotlib.
        Each feature is named "V<i>", with i's pulled in order of
        feature appearance.
        
        Returns:
            a list
        
        """
        return ['V%d' % (i+1) for i in range(self._imp_history.shape[1])]

    @property
    def _shadow_names(self):
        """Default shadow variable names
        
        Returns:
            a list
        
        """
        return ['shadowMin', 'shadowMean', 'shadowMax']
