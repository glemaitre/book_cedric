# %% [markdown]
#
# # Famille des modèles paramétriques
#
# ## Modèles linéaires

# %% [markdown]
#
# ### Principe de la fonction de coût

# %% [markdown]
#
# On this notebook, we will have a deeper look to linear models and especially
# the concept of loss functions. We will reuse the previous regression problem
# where we wanted to model the relationship between the penguins' flipper
# length and their body mass.

# %%
import pandas as pd

data = pd.read_csv("../datasets/penguins_regression.csv")
data.head()

# %%
import seaborn as sns
sns.set_context("poster")

# %%
_ = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")

# %% [markdown]
#
# We observe that there is a reasonable linear relationship between the flipper length
# and the body mass. Here, our target to be predicted will be the body mass while the
# flipper length will be a feature.

# %%
X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

# %% [markdown]
#
# In the previous notebook, we used a <tt>LinearRegression</tt> from scikit-learn and
# show that we could learn the state of the model from the data when calling
# <tt>fit</tt> and use these states for prediction when calling the method
# <tt>predict</tt>.

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
_ = ax.legend()

# %% [markdown]
#
# In the previous notebook, we quickly mentioned that the linear regression model was
# minizing an error between the true target and the predicted target. This error is also
# known as loss function. The loss that is minimized in this case is known as the least
# squared error. This loss is defined as:
#
# $$
# loss = (y - \hat{y})^2
# $$
#
# that is
#
# $$
# loss = (y - X \beta)^2
# $$
#
# We can check what the loss look likes in practice:

# %%


def se_loss(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    return loss


# %%
import numpy as np

xmin, xmax = -2, 2
xx = np.linspace(xmin, xmax, 100)

# %%
import matplotlib.pyplot as plt

plt.plot(xx, se_loss(0, xx), label="SE loss")
_ = plt.legend()

# %% [markdown]
#
# Looking at the shape of the loss function, we see that the bell shape of the loss will
# impact greatly the large error. In practice, this will have an impact on the fit.

# %%
data = data.append(
    {"Flipper Length (mm)": 230, "Body Mass (g)": 300},
    ignore_index=True
)
data

# %%
_ = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")

# %%
X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

# %%
import numpy as np
sample_weight = np.ones_like(y)
sample_weight[-1] = 10

# %%
model.fit(X, y, sample_weight=sample_weight)
y_pred = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
_ = ax.legend()

# %% [markdown]
#
# Instead of using the squared loss, we will use a loss known as the Huber loss. In this
# regard, we will use the HuberRegressor model available in scikit-learn. We will fit
# this model in the exact similar way that we previously did.

# %%
from sklearn.linear_model import HuberRegressor

model = HuberRegressor()
model.fit(X, y, sample_weight=sample_weight)
y_pred = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
_ = ax.legend()

# %% [markdown]
#
# We observe that the outlier has much less weight than in the case of the least squared
# loss.

# %%
def huber_loss(y_true, y_pred, *, epsilon):
    mask_greater_epsilon = np.abs(y_true - y_pred) > epsilon
    loss = np.zeros_like(y_pred)
    loss[mask_greater_epsilon] = np.abs(y_true - y_pred)[mask_greater_epsilon]
    loss[~mask_greater_epsilon] = se_loss(y_true, y_pred)[~mask_greater_epsilon]
    return loss

# %%
def absolute_loss(y_true, y_pred):
    loss = np.abs(y_true - y_pred)
    return loss

# %%
plt.plot(xx, se_loss(0, xx), label="SE loss")
plt.plot(xx, huber_loss(0, xx, epsilon=1), label="Huber loss")
plt.plot(xx, absolute_loss(0, xx), label="Absolute loss", linestyle="--")
plt.ylabel("Loss")
plt.xlabel("xx")
_ = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# %% [markdown]
#
# We observe that the Huber and absolute losses are penalizing less outliers. It means
# that these outliers will be less attractive and we will not try to find $\beta$ that
# try to minimize this large error. Indeed, the <tt>HuberRegressor</tt> will give an
# estimator of the median instead of the mean.
#
# If one is interesting in other quantile than the median, scikit-learn provides an
# estimator called `QuantileRegressor` that minimizes the pinball loss and provide a
# estimator of the requested quantile. For instance, one could request the median in the
# following manner:

# %%
from sklearn.linear_model import QuantileRegressor

model = QuantileRegressor(quantile=0.5)
model.fit(X, y, sample_weight=sample_weight)
y_pred = model.predict(X)

# %%
ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
_ = ax.legend()

# %% [markdown]
#
# ### Principe de la régularisation
#
# In this first example, we will show a known issue due to correlated features when
# fitting a linear model.
#
# The data generative process to create the data is a linear relationship between the
# features and the target. However, out of 5 features, only 2 features will be used
# while 3 other features will not be linked to the target. In addition, a little bit of
# noise will be added. When generating the dataset, we can as well get the true model.

# %%
from sklearn.datasets import make_regression

data, target, coef = make_regression(
    n_samples=2_000,
    n_features=5,
    n_informative=2,
    shuffle=False,
    coef=True,
    random_state=0,
    noise=30,
)

# %%
import seaborn as sns
sns.set_context("poster")

# %%
import pandas as pd

feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(coef, index=feature_names)
coef.plot.barh()
coef

# %% [markdown]
#
# Plotting the true coefficients, we observe that only 2 features out of the 5 features
# as an impact on the target.
#
# Now, we will fit a linear model on this dataset.

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(data, target)
linear_regression.coef_

# %%
feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()

# %% [markdown]
#
# We observe that we can recover almost the true coefficients. The small fluctuation are
# due to the noise that we added into the dataset when generating it.

# %%
import numpy as np

data = np.concatenate([data, data[:, [0, 1]], data[:, [0, 1]]], axis=1)

# %%
linear_regression = LinearRegression()
linear_regression.fit(data, target)
linear_regression.coef_

# %%
feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()

# %% [markdown]
#

# %% [markdown]
#
# ### Importance du prétraitement
#
# ### Alternative à la minimization moindres carrés
#
# ## Résolution de problèmes non-linéaires
#
# ## Quantifier l'incertitude des prédictions
