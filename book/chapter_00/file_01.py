# %% [markdown]
#
# # Famille des modèles paramétriques
#
# Dans le chapitre précédent, nous avons détaillé le principe d'un modèle
# prédictif de manière mathématique. Nous pouvons rappeler cette formulation :
#
# $$
# \hat{y} = f(X)
# $$
#
# Nous avons même donné un exemple d'un modèle assez naif ou nous avons utilisé
# une relation entre notre variable d'entrée et notre variable de sortie. Cette
# manière de définir un modèle est nommée **modèle paramétrique**. En effet,
# nous avons défini un modèle paramétrique de la forme suivante :
#
# $$
# \hat{y} = f(X) = X \beta
# $$
#
# Le paramètre $\beta$ est donc le **paramètre** de notre modèle. L'idée
# derrière cette famille de modèles est donc de pouvoir compresser
# l'information de notre jeu de données d'apprentissage avec seulement quelques
# paramètres et un *apriori* sur la relation entre $X$ et $y$ (nous reviendrons
# plus en détail sur cet aspect dans les sections qui viennent).
#
# Dans ce chapitre, nous allons tout d'abord détailler une des familles les
# plus simple : les modèles linéaires. Nous présenterons certaines composantes
# importantes de ce type de modèle. Par la suite, nous montrerons que ces
# modèles peuvent également utilisés pour des problèmes non-linéaires.
#
# ## Modèle linéaire
#
# Un modèle linéaire est un modèle paramétrique qui est défini par une relation
# entre $X$ et $y$ tel que $y$ est une combination linéaire de $X$. Notre
# modèle dans le chapitre précédent était un modèle linéaire :
#
# $$
# \hat{y} = f(X, \beta) = X \beta
# $$
#
# Le terme "combination" nous indique que nous pouvons généraliser cette
# relation en combinant toutes les variables d'entrée (i.e. colonnes) de $X$.
# Un tel modèle est donc défini de la manière suivante :
#
# $$
# \hat{y} = f(X, \beta) = X \beta = \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n
# X_n
# $$
#
# where $\beta_n$ est le paramètre associé à la variable $X_n$. La relation
# ci-dessus force notre modèle de prédire 0 lorsque les valeurs dans $X$ sont
# également à 0. Pour avoir plus de flexibilité, un paramètre $\beta_0$ est
# utilisé pour représenter cette constante et est appelé l'**intercept**.
#
# $$
# \hat{y} = f(X, \beta) = X \beta = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... +
# \beta_n X_n
# $$
#
# Nous pouvons donc comprendre que nous faisons un *apriori* entre le lien
# entre $X$ et $y$ : nous pensons que $y$ est une combinaison linéaire de $X$
# et que cet relations est suffisante. Un peu plus tard dans ce chapitre, nous
# donnerons des exemples où ce n'est pas le cas et où nous devrons modifier la
# formulation de notre modèle.
#
# En revanche, si cet *apriori* est correct, nous venons donc de compresser
# notre de dataset de taille composé de $N$ échantillons à un modèle de taille
# $P + 1$ paramètres (i.e. + 1 corresponds à l'intercept). Maintenant que nous
# avons défini notre modèle, il nous est possible de trouver les paramètres.
#
# ### Trouver le meilleur modèle possible
#
# Maintenant que nous connaisons la paramétrisation de notre modèle, nous
# pouvons l'illustrer sur le même jeu de données que nous avons utilisé dans
# le chapitre précédent. Tout d'abord, nous chargeons les données.

# %%
import pandas as pd

donnees = pd.read_csv("../datasets/penguins_regression.csv")
X = donnees[["Longueur Aileron (mm)"]]
y = donnees["Masse Corporelle (g)"]

# %% [markdown]
#
# Et nous pouvons visualaliser la relation entre $X$ et $y$ :

# %%
import seaborn as sns

sns.set_context("poster")
ax = donnees.plot.scatter(x=donnees.columns[0], y=donnees.columns[1])
_ = ax.set_title("Masse corporelle en fonction de\nla longueur d'aileron")

# %% [markdown]
#
# Il existe une infinité de modèle linéaire qui pourraient être utilisés pour
# pour prédire la masse corporelle de nos pingouins. Définissons une fonction
# Python générique qui permet de prédire la masse corporelle de notre pinguoin
# en fonction de la longueur d'aileron.


# %%
def modele_lineaire(longueur_aileron, parametres):
    # notre modèle est défini par: y = beta_0 + x_1 * beta_1
    return parametres[0] + parametres[1] * longueur_aileron


# %% [markdown]
#
# Maintenant que nous avons notre modèle, nous pouvons visualiser quelques
# modèles avec différents paramètres.

# %%
predictions_modele_1 = modele_lineaire(X, [-3_000, 30])
predictions_modele_2 = modele_lineaire(X, [-6_000, 50])
predictions_modele_3 = modele_lineaire(X, [-2_000, 30])

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
ax.plot(X, predictions_modele_1, label="Modèle #1")
ax.plot(X, predictions_modele_2, label="Modèle #2")
ax.plot(X, predictions_modele_3, label="Modèle #3")
ax.legend()
_ = ax.set_title("Quel modèle est le meilleur?")

# %% [markdown]
#
# A partir de ce graphique, la question que nous pourrions avoir est de savoir
# quel modèle est le meilleur. Qualitativement, nous pourrions dire que le
# modèle #1 est le pire modèle. Entre le modèle #2 et #3, nous pourrions
# préviligier le modèle #2 car il semble plus "centré" avec nos données.
#
# Cependant, choisir un modèle ne peut-être basé sur une évaluation
# qualitative. Dans le chapitre précédent, nous avons utilisé différentes
# méthodes qui calculaient une erreur. Nous pouvons ici calculer une erreur
# donnée : l'erreur quadratique moyenne.

# %%
from sklearn.metrics import mean_squared_error

print(f"Erreur du modèle #1: {mean_squared_error(y, predictions_modele_1):.2f}")
print(f"Erreur du modèle #2: {mean_squared_error(y, predictions_modele_2):.2f}")
print(f"Erreur du modèle #3: {mean_squared_error(y, predictions_modele_3):.2f}")

# %% [markdown]
#
# En utilisant cette erreur, nous avons la confirmation que le modèle #2 a la
# plus petite erreur. En revanche, est ce que ce modèle est le meilleur
# possible ? Si non, comment pouvons nous trouver un modèle la plus faible
# possible ?
#
# Nous donc un probème d'optimisation où nous voudrions minimiser cette erreur
# également appelée **fonction de coût** dans ce contexte. Donc, nous pouvons
# donc définir notre fonctionde coup comme l'erreur quadratique moyenne
# formulée ci-dessous :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(X_i, \beta)
# \right)^2
# $$
#
# Et nour chercherons donc à minimiser cette fonction de coût. En d'autre
# termes, nous serions intéressés par trouver le minimum de
# $\mathcal{L}(\beta)$.
#
# $$
# \min_{\beta} \mathcal{L}(\beta)
# $$
#
# Trouver le minimum d'une fonction donnée est un problème typique
# d'**optimisation methématique** et il existe plusieurs méthodes, certaines
# plus performantes que d'autres, dépendant de la fonction à minimiser. Nous
# pouvons mentionner les méthodes basées sur le gradient qui nécessitent de
# pouvoir dériver la fonction de coût.
#
# ```{note}
# Il existe une solution analytique pour la fonction de coût que nous avons
# définie.
#
# $$
# \beta = \left( X^T X \right)^{-1} X^T y
# $$
#
# En revanche, nous avons introduit la méthode de gradient car elle nous
# permettra d'avoir une certaine réflexion concernant les futurs fonctions de
# coût que nous allons définir.
# ```
#
# Nous avons la chance que notre fonction de coût définie comme l'erreur
# quadratique moyenne soit facilement dérivable. Il serait donc facile de
# calculer le mimimum de cette fonction de coût.
#
# `scikit-learn` nous propose une classe dénommée `LinearRegression` qui
# permet de minimser cette fonction de coût. Nous allons la mettre en pratique
# dès maintenant.

# %%
from sklearn.linear_model import LinearRegression

modele = LinearRegression()
modele.fit(X, y)

# %% [markdown]
#
# Maintenant que notre modèle est entrainé, nous pouvons l'utiliser pour
# observer visuellement quels sont les prédictions produites par ce modèle.

# %%
predictions = modele.predict(X)

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
ax.plot(X, predictions, label="Regression lineaire")
ax.legend()
_ = ax.set_title("Modele LinearRegression")

# %% [markdown]
#
# Ce modèle semble donc bien minimiser la fonction de coût et est
# qualititivement correct. Nous pouvons donc maintenant nous pencher sur notre
# modèle et obtenir la valeurs des paramètres.

# %%
modele.coef_

# %%
modele.intercept_

# %% [markdown]
#
# Nous pouvons donc observer deux attributs qui correspondent aux paramètres de
# notre modèle. `coef_` contient les paramètres de $\beta_1, ..., \beta_n$
# alors que `intercept_` contient le paramètre de $\beta_0$.
#
# ```{note}
# Dans scikit-learn, les attributs finissant par `_` sont des attributs
# qui sont créés après avoir appelé la méthode `fit()`. Ils sont liés à
# l'algorithme d'apprentissage et seront nécessaires pour pouvoir créer
# des prédictions.
# ```
#
# Maintenant, nous pouvons interpréter la valeurs des paramètres de notre
# modèle. La valeur dans la variable `coef_` est la valeur associé à la
# variable "Longueur Aileron (mm)". Cette valeur correspond à l'incrément de la
# masse corporelle lorsqu'un pingouin à un incrément de 1 mm de longueur
# d'aileron. Dans notre cas, cette valeur est d'environ 50 grammes. La valeur
# de la variable `intercept_` correspond à la valeur de l'ordonnée à l'origine
# : un pinguoin avec un aileron de 0 mm aura une masse corporelle de -5781
# grammes! Cette valeur est beaucoup plus compliquée à comprendre mais elle
# nous permet d'avoir un modèle plus flexible, ne passant pas par l'origine.

# # %%
# _ = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")

# # %% [markdown]
# #
# # We observe that there is a reasonable linear relationship between the flipper
# # length and the body mass. Here, our target to be predicted will be the body
# # mass while the flipper length will be a feature.

# # %%
# X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

# # %% [markdown]
# #
# # In the previous notebook, we used a <tt>LinearRegression</tt> from scikit-learn and
# # show that we could learn the state of the model from the data when calling
# # <tt>fit</tt> and use these states for prediction when calling the method
# # <tt>predict</tt>.

# # %%
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X, y)
# y_pred = model.predict(X)

# # %%
# ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
# ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
# _ = ax.legend()

# # %% [markdown]
# #
# # In the previous notebook, we quickly mentioned that the linear regression model was
# # minizing an error between the true target and the predicted target. This error is also
# # known as loss function. The loss that is minimized in this case is known as the least
# # squared error. This loss is defined as:
# #
# # $$
# # loss = (y - \hat{y})^2
# # $$
# #
# # that is
# #
# # $$
# # loss = (y - X \beta)^2
# # $$
# #
# # We can check what the loss look likes in practice:

# # %%


# def se_loss(y_true, y_pred):
#     loss = (y_true - y_pred) ** 2
#     return loss


# # %%
# import numpy as np

# xmin, xmax = -2, 2
# xx = np.linspace(xmin, xmax, 100)

# # %%
# import matplotlib.pyplot as plt

# plt.plot(xx, se_loss(0, xx), label="SE loss")
# _ = plt.legend()

# # %% [markdown]
# #
# # Looking at the shape of the loss function, we see that the bell shape of the loss will
# # impact greatly the large error. In practice, this will have an impact on the fit.

# # %%
# data = data.append(
#     {"Flipper Length (mm)": 230, "Body Mass (g)": 300},
#     ignore_index=True
# )
# data

# # %%
# _ = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")

# # %%
# X, y = data[["Flipper Length (mm)"]], data["Body Mass (g)"]

# # %%
# import numpy as np
# sample_weight = np.ones_like(y)
# sample_weight[-1] = 10

# # %%
# model.fit(X, y, sample_weight=sample_weight)
# y_pred = model.predict(X)

# # %%
# ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
# ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
# _ = ax.legend()

# # %% [markdown]
# #
# # Instead of using the squared loss, we will use a loss known as the Huber loss. In this
# # regard, we will use the HuberRegressor model available in scikit-learn. We will fit
# # this model in the exact similar way that we previously did.

# # %%
# from sklearn.linear_model import HuberRegressor

# model = HuberRegressor()
# model.fit(X, y, sample_weight=sample_weight)
# y_pred = model.predict(X)

# # %%
# ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
# ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
# _ = ax.legend()

# # %% [markdown]
# #
# # We observe that the outlier has much less weight than in the case of the least squared
# # loss.

# # %%
# def huber_loss(y_true, y_pred, *, epsilon):
#     mask_greater_epsilon = np.abs(y_true - y_pred) > epsilon
#     loss = np.zeros_like(y_pred)
#     loss[mask_greater_epsilon] = np.abs(y_true - y_pred)[mask_greater_epsilon]
#     loss[~mask_greater_epsilon] = se_loss(y_true, y_pred)[~mask_greater_epsilon]
#     return loss

# # %%
# def absolute_loss(y_true, y_pred):
#     loss = np.abs(y_true - y_pred)
#     return loss

# # %%
# plt.plot(xx, se_loss(0, xx), label="SE loss")
# plt.plot(xx, huber_loss(0, xx, epsilon=1), label="Huber loss")
# plt.plot(xx, absolute_loss(0, xx), label="Absolute loss", linestyle="--")
# plt.ylabel("Loss")
# plt.xlabel("xx")
# _ = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# # %% [markdown]
# #
# # We observe that the Huber and absolute losses are penalizing less outliers. It means
# # that these outliers will be less attractive and we will not try to find $\beta$ that
# # try to minimize this large error. Indeed, the <tt>HuberRegressor</tt> will give an
# # estimator of the median instead of the mean.
# #
# # If one is interesting in other quantile than the median, scikit-learn provides an
# # estimator called `QuantileRegressor` that minimizes the pinball loss and provide a
# # estimator of the requested quantile. For instance, one could request the median in the
# # following manner:

# # %%
# from sklearn.linear_model import QuantileRegressor

# model = QuantileRegressor(quantile=0.5)
# model.fit(X, y, sample_weight=sample_weight)
# y_pred = model.predict(X)

# # %%
# ax = data.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
# ax.plot(X, y_pred, label=model.__class__.__name__, color="black", linewidth=4)
# _ = ax.legend()

# # %% [markdown]
# #
# # ### Principe de la régularisation
# #
# # In this first example, we will show a known issue due to correlated features when
# # fitting a linear model.
# #
# # The data generative process to create the data is a linear relationship between the
# # features and the target. However, out of 5 features, only 2 features will be used
# # while 3 other features will not be linked to the target. In addition, a little bit of
# # noise will be added. When generating the dataset, we can as well get the true model.

# # %%
# from sklearn.datasets import make_regression

# data, target, coef = make_regression(
#     n_samples=2_000,
#     n_features=5,
#     n_informative=2,
#     shuffle=False,
#     coef=True,
#     random_state=0,
#     noise=30,
# )

# # %%
# import seaborn as sns
# sns.set_context("poster")

# # %%
# import pandas as pd

# feature_names = [f"Features {i}" for i in range(data.shape[1])]
# coef = pd.Series(coef, index=feature_names)
# coef.plot.barh()
# coef

# # %% [markdown]
# #
# # Plotting the true coefficients, we observe that only 2 features out of the 5 features
# # as an impact on the target.
# #
# # Now, we will fit a linear model on this dataset.

# # %%
# from sklearn.linear_model import LinearRegression

# linear_regression = LinearRegression()
# linear_regression.fit(data, target)
# linear_regression.coef_

# # %%
# feature_names = [f"Features {i}" for i in range(data.shape[1])]
# coef = pd.Series(linear_regression.coef_, index=feature_names)
# _ = coef.plot.barh()

# # %% [markdown]
# #
# # We observe that we can recover almost the true coefficients. The small fluctuation are
# # due to the noise that we added into the dataset when generating it.

# # %%
# import numpy as np

# data = np.concatenate([data, data[:, [0, 1]], data[:, [0, 1]]], axis=1)

# # %%
# linear_regression = LinearRegression()
# linear_regression.fit(data, target)
# linear_regression.coef_

# # %%
# feature_names = [f"Features {i}" for i in range(data.shape[1])]
# coef = pd.Series(linear_regression.coef_, index=feature_names)
# _ = coef.plot.barh()

# # %% [markdown]
# #

# # %% [markdown]
# #
# # ### Importance du prétraitement
# #
# # ### Alternative à la minimization moindres carrés
# #
# # ## Résolution de problèmes non-linéaires
# #
# # ## Quantifier l'incertitude des prédictions
