# %% [markdown]
#
# # Introduction
#
# ## Motivations de l'apprentissage automatique
#
# - Créer une fonction de prédiction y = f(X)
# - Définir y, X et f()

# %% [markdown]
#
# We will start by introducing the concept of linear regression.
# First, we will not use any advanced tool as scikit-learn and instead made use
# of only NumPy. First, let's load the dataset that we will use for this
# exercise.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")
data.head()

# %% [markdown]
#
# We will use a dataset containing records of some penguins flipper length and
# body mass. The first task that we want to accomplish is to learn a predictive
# model where we would like to estimate the body mass of a penguins given its
# flipper length. Since the target is continuous, this problem is a regression
# problem.
#
# Let's first have a look at the relationship between these measurements

# %%
import seaborn as sns
sns.set_context("poster")

# %%
ax = data.plot.scatter(x=data.columns[0], y=data.columns[1])
_ = ax.set_title("Can I predict penguins' body mass?")

# %% [markdown]
#
# We see could see that we have a linear trend between the flipper length and
# the body mass: longer is the penguin's flipper, heavier is the penguin. The
# first predictive model that we would like to have will model the relationship
# between these two variables as a linear relationship.
#
# Thus, in this example:
#
# - the flipper length is a feature. Here, we will only use a single feature.
#   In practice, we will have many features when trying to create a predictive
#   model.
# - the body mass is the variable that we would like to predict, thus it is the
#   target.
#
# A pair of measurement (flipper length, body mass) is called a sample. We
# learn a predictive model from available pair of such features/target. At
# prediction time, we will only have the features available and the goal is to
# predict the potential target. To evaluate a predictive model, we can then use
# some of the features and compare the predictions given by the model with the
# true target.
#
# For the rest of this lecture, we will denote the variable `X` as the matrix
# of shape `(n_samples, n_features)` and the target will be denoted by the
# variable `y` as a vector of shape `(n_samples,)`.

# %%
X, y = data[["Flipper Length (mm)"]], data[["Body Mass (g)"]]

# %% [markdown]
#
# ## Example introductif à l'interface de programmation de `scikit-learn`
#
# - Formuler le problème de least squares
# - Résoudre avec NumPy
# - Notre propre classe et parallèle avec `scikit-learn`

# %% [markdown]
#
# To start, we would like to model the relationship between `X` and `y` by a
# linear relationship. Thus, we can formalize it as:
#
# $$
# y = X \beta
# $$
#
# where $\beta$ are the coefficients of our linear model. We could expand this
# equation for all available features as:
#
# $$
# y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n
# $$
#
# Here, we only have a single feature, $\beta_1$, that is the flipper length.

# Finding a linear model is equivalent to find the "best possible" $\beta$.
# What is the best $\beta$? The best $\beta$ would be the $\beta$ that once
# used with $X$ will predict $\hat{y}$ such that the $\hat{y}$ (i.e. the
# predicted target) is as closed as possible of $y$ (i.e. the true target).
# Thus, we would like to minimize an error. We can find the $\beta$ from the
# equation above:

# $$
# X^T y = X^T X \beta
# $$

# and thus

# $$
# \beta = (X^T X)^{-1} X^T y
# $$

# %%
X["Intercept"] = 1
X.head()

# %%
import numpy as np

coef = np.linalg.inv(X.T @ X) @ X.T @ y
coef

# %%
y_pred = np.dot(X, coef)
y_pred[:5]

# %%
ax = data.plot.scatter(x=data.columns[0], y=data.columns[1])
ax.plot(X["Flipper Length (mm)"], y_pred, color="black", linewidth=4)
_ = ax.set_title("Can I predict penguins' body mass")

# %% [markdown]
#
# ## Evaluation d'un modèle prédictif
#
# - Evaluation sur un seul jeu de donnée
# - Définition de l'erreur empirique et de généralisation
# - Définition de sous- et sur-apprentissage
# - Evaluation sur un set d'entrainement et de test
# - Introduction à la cross-validation

# %% [markdown]
#
# Scikit-learn is using Python classes to give some state to Python object. In
# machine-lerning context, it is handy to have object that will "learn" some
# internal states. Once these states are learnt, the object could be used to
# make the prediction. In scikit-learn, we thus have Python class to create
# instance that will have:
#
# - a `fit` method to learn the internal states
# - a `predict` method to output some predicted values given some input data

# %%


class LinearRegression:

    def __init__(self, intercept=True):
        self.intercept = intercept

    def fit(self, X, y):
        if self.intercept:
            X = np.hstack(
                [X, np.ones((X.shape[0], 1))]
            )
        self.coef_ = coef = np.linalg.inv(X.T @ X) @ X.T @ y
        self._target_name = y.columns
        return self

    def predict(self, X):
        if self.intercept:
            X = np.hstack(
                [X, np.ones((X.shape[0], 1))]
            )
        return pd.DataFrame(
            np.dot(X, coef), columns=self._target_name
        )


# %%
model = LinearRegression(intercept=False)
model.fit(X, y).predict(X)

# %%
model.coef_

# %%
from sklearn.linear_model import LinearRegression

X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
y_pred[:5]

# %%
ax = data.plot.scatter(x=data.columns[0], y=data.columns[1])
ax.plot(X["Flipper Length (mm)"], y_pred, color="black", linewidth=4)
_ = ax.set_title("Can I predict penguins' body mass")

# %% [markdown]
#
# A scikit-learn predictive model will store the state of the model with
# attribute that finishes with an underscore. They are fitted attributes. For
# our linear model, they will be coef_ and intercept_.

# %%
model.coef_, model.intercept_

# %% [markdown]
#
# As previously stated, we can compute a metric to evaluate how good our
# trained model is.

# %%
from sklearn.metrics import r2_score

r2_score(y, model.predict(X))

# %% [markdown]
#
# However, there is something wrong in what we just did. Indeed, if we would
# have a predictive model that would have memorize the training set, we would
# have obtain a perfect score. In practice, we don't use the same dataset to
# train and test a model to get a true estimate of the capability of a model to
# predict targets on new dataset. A metric computed on the training set is also
# called empirical error while the error computed on the testing set is called
# generalization error.
#
# Thus, we could have first split our dataset into two sets: a training set to
# train our model and a testing set to check the statistical performance of our
# model.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# %%
model.fit(X_train, y_train)

# %%
model.coef_, model.intercept_

# %%
model.coef_, model.intercept_

# %%
r2_score(y_test, model.predict(X_test))

# %% [markdown]
#
# We observe that our model is not as precise on the testing set than on the
# training set. But this is not catastrophic. We can show with another type of
# a model (decision tree) that the different can be quite drastic.

# %%
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# %%
r2_score(y_train, model.predict(X_train))

# %%
r2_score(y_test, model.predict(X_test))

# %% [markdown]
#
# Let's come back to our linear model trained and tested on distinct sets. We
# observe a small difference between the training and testing scores. However,
# we are unable to know if the difference is significant, meaning that the
# difference might only be due to some random fluctuation given by our random
# initial data split. A proper evaluation should provide an estimate of the
# distribution of the different scores and not only a point-estimate.
# Cross-validation is a framework that would allow to take into account such
# variation.
#
# The idea between cross-validation is to repeat the evaluation by varying the
# train and test set. This evaluation will therefore take into account the
# fluctuation that could happen in the "fit" process as well as in the
# "predict" process.

# %%
model = LinearRegression()

# %% [markdown]
#
# Scikit-learn provides the sklearn.model_selection.cross_validate function to
# repeat this train-test evaluation.

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results = cross_validate(
    model, X, y, cv=cv,
    scoring="r2",
    return_train_score=True,
)

# %%
cv_results = pd.DataFrame(cv_results)

# %%
cv_results[["train_score", "test_score"]]

# %%
cv_results[["train_score", "test_score"]].mean()

# %%
cv_results[["train_score", "test_score"]].std()

# %% [markdown]
#
# We can observe that we have close result between the train and test score.
# However, the distribution of the test score is a bit larger. We could indeed
# use a repeated k-fold cross-validation to get more estimate and plot the
# distributions of the scores.

# %%
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_repeats=10, n_splits=3, random_state=0)
cv_results = cross_validate(
    model, X, y, cv=cv,
    scoring="r2",
    return_train_score=True
)
cv_results = pd.DataFrame(cv_results)

# %%
ax = cv_results[["train_score", "test_score"]].plot.hist(alpha=0.7)
ax.set_xlim([0, 1])

# %% [markdown]
#
# Visually, we can see that our model is behaving similarly on the training and
# testing sets with quite a small variation. This is quite a good new. Indeed,
# we can trust these results.
