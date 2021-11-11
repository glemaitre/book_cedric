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
# Et nous chercherons donc à minimiser cette fonction de coût. En d'autre
# termes, nous serions intéressés par trouver le minimum de
# $\mathcal{L}(\beta)$. Cette minimisation est également appellée méthode
# des moindres carrés.
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
ax.plot(X, predictions, label="Regression linéaire")
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
#
# ### Alternative à la méthode des moindres carrés
#
# La méthode des moindres carrés est la méthode la plus simple que nou pouvions
# présenter. En revanche, cette méthode à des limitations connues. Nous allons
# en présenter l'une d'entre elles, ainsi montrer comment nous pouvons la
# contourner.
#
# Nous allons reprendre les mêmes données que précédemment utilisées. En
# revanche, nous allons simuler que des erreurs de saisie sont survenues lors
# de la collecte des données. Pour cela, nous allons rajouter un échantillon de
# pinguoin qui aura un aileron de longueur de 230 mm et une masse corporelle de
# 300 grammes. Cette erreur pourrait être dûe à une erreur de saisie ou un 0
# est manquant.

# %%
donnees = donnees.append(
    {"Longueur Aileron (mm)": 230, "Masse Corporelle (g)": 300}, ignore_index=True
)
donnees.tail()
X = donnees[["Longueur Aileron (mm)"]]
y = donnees["Masse Corporelle (g)"]

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
_ = ax.set_title("Nos données avec une\nerreur de saisie")

# %% [markdown]
#
# Nous pouvons observer que nous avons un nouveau échantillon dans le cadrant
# en bas à droite de notre graphique. Nous allons maintenant entraîner un
# modèle linéaire qui minimise l'erreur quadratique moyenne. Pour simuler que
# nous avons plusieurs erreurs de saisie, nous allons entraîner notre modèle en
# assignant des poids à chaque échantillon : tous les échantillons auront un
# poids de 1, sauf le nouvel échantillon qui aura un poids de 10.

# %%
import numpy as np

poids = np.ones(y.size)
poids[-1] = 10

# %%
modele = LinearRegression()
predictions_err_quadratique = modele.fit(X, y, sample_weight=poids).predict(X)

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
ax.plot(X, predictions_err_quadratique, label="Regression linéaire")
ax.legend()
_ = ax.set_title("Modele LinearRegression")

# %% [markdown]
#
# Nous pouvons donc observer que le faite d'avoir des erreurs de saisie à une
# influence non négligeable sur la qualité de notre modèle. Ceci peut-être
# expliqué par le type de fonction de coût que nous utilisons. Il sera plus
# facile d'obtenir une intuition en représentant graphiquement cette fonction.


# %%
def erreur_quadratique(cible, prediction):
    cout = (cible - prediction) ** 2
    return cout


# %%
import matplotlib.pyplot as plt

xmin, xmax = -3, 3
xx = np.linspace(xmin, xmax, 100)

plt.plot(xx, erreur_quadratique(0, xx), label="Erreur quadratique")
_ = plt.legend()

# %% [markdown]
#
# Nous pouvons donc observer que le faite d'élever au carré l'erreur pénalise
# extrêment les échantillons avec une grande erreur. Il serait donc intéressant
# d'utiliser une fonction de coût qui affectera un coût moindre aux
# échantillons pour lesquel notre modèle commet le plus d'erreur. Au lieu de
# prendre le carré de l'erreur, nous pourrions seulement utiliser la valeur
# absolue de l'erreur. Cette erreur sera donc l'**erreur absolute moyenne** et
# peut-être formulée comme suit :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N \left| y_i - f(X_i, \beta)
# \right|
# $$
#
# Nous pouvons comparer visualement la représentation de cette fonction de coût
# avec la représentation de la fonction d'erreur quadratique.


# %%
def erreur_absolue(cible, prediction):
    cout = abs(cible - prediction)
    return cout


# %%
plt.plot(xx, erreur_quadratique(0, xx), label="Erreur quadratique")
plt.plot(xx, erreur_absolue(0, xx), label="Erreur absolue")
plt.title("Comparaison des fonctions de coût")
_ = plt.legend()

# %% [markdown]
#
# On peut donc observer que l'erreur absolue pénalisera les échantillons avec
# une grande erreur. En revanche, si nous nous attardons sur l'erreur aboslue
# nous pouvons observer quelle n'est pas dérivable en 0. Ceci nous empêche
# d'utiliser une méthode d'optimisation basée sur le gradient ce qui est
# problématique.
#
# Si nous voulons utiliser une méthode par descente de gradient, il est donc
# nécessaire de trouver un moyen de combiner les deux fonctions de coût :
# utiliser l'erreur absolue loin de 0 pour moins pénaliser les échantillons
# avec une grande erreur et utiliser l'erreur quadratique quand l'erreur est
# proche de 0 pour que nous puissions déterminer le gradient.
#
# Cette fonction de coût est connu sous le nom de la fonction de Huber et est
# formulée comme suit :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum^{N}_{i=1} \begin{cases}
# \left( y_i - f(X_i, \beta) \right)^2 & \text{si } \left|
# y_i - f(X_i, \beta) \right| < \epsilon \\
# 2 \epsilon \left( | y_i - f(X_i, \beta) | - \epsilon^2 \right) & \text{sinon}
# \end{cases}
# $$


# %%
def fonction_huber(cible, prediction, *, epsilon=1.35):
    cout_absolue = erreur_absolue(cible, prediction)
    cout_quadratique = erreur_quadratique(cible, prediction)

    plus_grand_epsilon = cout_absolue > epsilon

    cout = np.zeros_like(prediction)
    cout[~plus_grand_epsilon] = cout_quadratique[~plus_grand_epsilon]
    cout[plus_grand_epsilon] = (
        2 * epsilon * cout_absolue[plus_grand_epsilon] - epsilon ** 2
    )
    return cout


# %%
plt.plot(xx, erreur_quadratique(0, xx), label="Erreur quadratique")
plt.plot(xx, erreur_absolue(0, xx), label="Erreur absolue")
plt.plot(
    xx, fonction_huber(0, xx, epsilon=1.0), label="Fonction de Huber", linestyle="--"
)
plt.title("Comparaison des fonctions de coût")
_ = plt.legend()

# %% [markdown]
#
# Nous pouvons donc observer que la fonction de Huber a les avantages des deux
# fonctions de coût précédentes. `scikit-learn` propose une classe appelée
# `HuberRegressor` qui permettra d'optimiser cette fonction de coût. Nous
# allons donc utiliser ce modèle sur notre jeu de données et observer la
# différence sur les prédictions.

# %%
from sklearn.linear_model import HuberRegressor

modele = HuberRegressor()
predictions_err_huber = modele.fit(X, y, sample_weight=poids).predict(X)

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
ax.plot(X, predictions_err_quadratique, label="Quadratique")
ax.plot(X, predictions_err_huber, label="Huber")
ax.legend()
_ = ax.set_title("Comparaison modèle linéaire")

# %% [markdown]
#
# Nous pouvons donc constater que le modèle linéaire minimisant la fonction de
# Huber permet d'obtenir un meilleur modèle que celui minimisant la fonction
# de coût quadratique.
#
# Pour confirmer de manière quantitative, nous pourrions calculer des erreurs.

# %%
from sklearn.metrics import mean_absolute_error

print(
    "Modèle linéaire quadratique:\n"
    "  Erreur quadratique moyenne : "
    f"{mean_squared_error(y, predictions_err_quadratique):.2f}\n"
    "  Erreur absolue moyenne : "
    f"{mean_absolute_error(y, predictions_err_quadratique):.2f}"
)
print(
    "Modèle linéaire Huber:\n"
    "  Erreur quadratique moyenne : "
    f"{mean_squared_error(y, predictions_err_huber):.2f}\n"
    "  Erreur absolue moyenne : "
    f"{mean_absolute_error(y, predictions_err_huber):.2f}"
)

# %% [markdown]
#
# Nous pouvons confirmer que le modèle linéaire quadratique a une erreur
# quadratique moins élevèe que le modèle linéaire Huber. En revanche, nous
# avons la conclusion opposée pour l'erreur absolue.
#
# Il est quand même intéressant de mentionner qu'il es possible d'optimiser la
# fonction de coût basée sur l'erreur absolue. En revanche, la méthode
# d'optimisation sera différente. Ce type de d'estimateur est connue en anglais
# sous le nom de "Least Absolute Deviation" (LAD). Cet estimateur minimisera
# donc la fonction de coût des erreurs absolues et sera un estimateur de la
# médiane de nos données. `scikit-learn` propose une classe appelée
# `QuantileRegressor` qui permet de regresser n'importe quelle quantile et
# notemment la médiane.

# %%
from sklearn.linear_model import QuantileRegressor

modele = QuantileRegressor(alpha=0.5, solver="highs")
predictions_err_absolue = modele.fit(X, y, sample_weight=poids).predict(X)

# %%
ax = donnees.plot.scatter(
    x=donnees.columns[0], y=donnees.columns[1], color="black", alpha=0.2
)
ax.plot(X, predictions_err_quadratique, label="Quadratique")
ax.plot(X, predictions_err_huber, label="Huber")
ax.plot(X, predictions_err_absolue, label="Absolue", linestyle="--")
ax.legend()
_ = ax.set_title("Comparaison modèle linéaire")

# %% [markdown]
#
# Nous pouvons donc constater que l'estimateur basé sur la fonction de coût de
# Huber est très proche de l'estimateur de la médiane. Nous reviendrons plus
# tard dans ce chapitre sur l'estimateur quantiles pour estimer des intervalles
# de confiance autour de la médiane.

# %% [markdown]
#
# ### Principe de la régularisation
#
# Dans la section précédente, nous avons abordé le principe de la fonction de
# coût. Nous avons étudié et analysé différentes types d'erreur. En revanche,
# nous n'avons pas abordé les différentes formes de régularisation.
#
# Dans cette section, nous allons dans un premier temps motivé l'utilité de la
# régularisation. Dans un deuxième temps, nous allons présenter les différents
# types de régularisation.
#
# Pour motiver l'utilité de la régularisation, nous allons générer un jeu de
# données simple que nous allons pouvoir facilement interpréter. Nous allons
# générer une cible à partir d'une combination linéaire de deux variables. Nous
# allons également ajouter trois variables qui ne seront pas prédictives (i.e.
# aucun lien n'existera entre la cible `y` et ces variables). Pour rendre le
# problème plus réaliste, nous allons ajouter un bruit Gaussien à la cible.
#
# La fonction `make_regression` dans `scikit-learn` nous permet de générer un
# tel jeu de données.

# %%
from sklearn.datasets import make_regression

n_features = 6
X, y, true_coef = make_regression(
    n_samples=1_000,
    n_features=n_features,
    n_informative=2,
    shuffle=False,
    coef=True,
    random_state=0,
    noise=30,
)

# %% [markdown]
#
# En plus des données `X` et `y`, cette fonction nous fournit également les
# coefficients de la combinaison linéaire. Nous pouvons visualiser ces
# coefficients.

# %%
nom_colonnes = [f"Colonne #{i}" for i in range(n_features)]
true_coef = pd.Series(true_coef, index=nom_colonnes)
true_coef.plot.barh()
true_coef

# %% [markdown]
#
# Nous pouvons également visualiser les liens marginales entre la cible `y` et
# chacune des variables composant notre jeu de données `X`.

# %%
donnees = pd.DataFrame(X, columns=nom_colonnes)
donnees["Cible"] = y
donnees = donnees.melt(id_vars="Cible", var_name="Variable", value_name="Valeur")

# %%
sns.lmplot(
    x="Valeur",
    y="Cible",
    col="Variable",
    hue="Variable",
    data=donnees,
    col_wrap=2,
    ci=None,
    scatter_kws={"s": 50, "alpha": 0.3},
)

# %% [markdown]
#
# Sur ce graphique, nous pouvons donc confirmer qu'il existe une lien entre les
# deux premières variables et notre cible alors qu'il n'existe aucun lien avec
# les autres variables.
#
# Maintenant, que nous avons quelques intuitions concernant notre jeu de
# données, nous allons créer un modèle linéaire pour estimer les coefficients
# de la combinaison linéaire. Mais tout d'abord, nous allons évaluer un tel
# modèle via une validation croisée. Nous utiliserons le score $R^2$ pour
# evaluer notre modèle (ce score est une mesure de la qualité du modèle où le
# maximum score est 1).

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

modele = LinearRegression()
cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
cv_results = cross_validate(
    modele, X, y, cv=cv, return_train_score=True, return_estimator=True
)
cv_results = pd.DataFrame(cv_results)

# %%
ax = cv_results[["train_score", "test_score"]].plot.hist(bins=10, alpha=0.5)
_ = ax.set_xlim([0, 1])

# %% [markdown]
#
# Nous pouvons donc constater que notre modèle nous permet d'obtenir de bonnes
# prédictions. Nous pouvons inspecter les coefficients des différents modèles
# obtenus pendant la validation croisée.

# %%
coef = pd.DataFrame(
    [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
    columns=nom_colonnes,
)
ax = coef.plot.box(vert=False, whis=10)
_ = ax.set_title("Coefficients des modèles linéaires \nobtenus par validation croisée")

# %% [markdown]
#
# Nous pouvons donc observer en utilisant une représentation graphique sous
# forme de boîte à moustache que les coefficients des modèles obtenus sont très
# proches de ceux de la combinaison linéaire.
#
# So far, so good! Nous allons mainteant introduire des **variables
# collinéaires**. Une variable collinéaire est une variable qui sera corrélé
# avec une autre. Nous allons analyser le type de problème engendré par ce type
# de variables lorsque un modèle linéaire utilisant les moindres carrés sera
# appliqué.
#
# Pour cela, nous allons répéter plusieurs fois les variables prédictives
# plusieurs fois.

# %%
X = np.concatenate([X, X[:, [0, 1]], X[:, [0, 1]]], axis=1)

# %% [markdown]
#
# Nous allons répéter l'expérience précédente et représenter graphiquement
# les coefficients des modèles obtenus.

# %%
cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
cv_results = cross_validate(
    modele, X, y, cv=cv, return_train_score=True, return_estimator=True
)

# %%
nom_colonnes = [f"Colonne #{i}" for i in range(X.shape[1])]
coef = pd.DataFrame(
    [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
    columns=nom_colonnes,
)
ax = coef.plot.box(vert=False, whis=1e15)
_ = ax.set_title("Coefficients des modèles linéaires \nobtenus par validation croisée")

# %% [markdown]
#
# Nous pouvons remarquer que les coefficients des modèles obtenus sont
# extrêmement élevés et complètement dissociés des valeurs originales. Ce
# problème est lié à une imprécision des calculs numériques.
#
# Nous pouvons aller un peu plus loin dans les détails en rappelant l'équation
# Normale pour résoudre notre système d'équations linéaires :
#
# $$
# \beta = (X^T X)^{-1} X^T y
# $$
#
# Dans cette équation, nous pouvons voir que nous devons inverser la matrice de
# Gram $(X^T X)$. Si cette matrice n'est pas inversible, nous serions donc dans
# l'impossibilité de calculer les coefficients du modèle. Une manière de
# vérifier est de calculer le déterminant de cette matrice.

# %%
np.linalg.det(X.T @ X)

# %% [markdown]
#
# Le déterminant étant nul, il n'est donc pas possible d'inverser la matrice :

# %%
try:
    np.linalg.inv(X.T @ X)
except np.linalg.LinAlgError as e:
    print(e)

# %% [markdown]
#
# En pratique, cette matrice n'utilise pas strictement la fonction inverse
# ci-dessus ce qui explique le réultat obtenu qui est cependant incorrect.
#
# Afin de résoudre ce problème, nous pouvons introduire le principe de
# régularisation où l'idée est d'ajouter à la fonction de coût un terme
# permettant de contraindre d'une manière donnée (i.e. qui dépend du type de
# régularisation) la valeur des coefficients.
#
# Nous allons tout d'abord formuler la régularisation de type L2 :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(X_i, \beta)
# \right)^2 + \alpha \|\beta\|_2^2
# $$
#
# where $\alpha$ est le paramètre controllant l'impact de la régularisation.
#
# En analysant la fonction de coût, nous pouvons remarquer que si le paramètre
# $\alpha$ est nul, le terme contraignant les coefficients sera nul et donc
# nous avons une simple régression linéaire. En revanche, si $\alpha$ est
# élevé, nous allons contraindre les coefficients plutôt que de réduire le
# terme correspondant à l'erreur quadratique. Donc nous obtiendrons un modèle
# qui minimisera la valeur des coefficients.
#
# Ce type de modèle est appelé **Ridge regression**. Nous allons le mettre de
# suite en oeuvre en utilisant `scikit-learn` en répétant l'expérience
# précédente.

# %%
from sklearn.linear_model import Ridge

modele = Ridge(alpha=1.0)
cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
cv_results = cross_validate(
    modele, X, y, cv=cv, return_train_score=True, return_estimator=True
)

# %%
nom_colonnes = [f"Colonne #{i}" for i in range(X.shape[1])]
coef = pd.DataFrame(
    [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
    columns=nom_colonnes,
)
ax = coef.plot.box(vert=False, whis=100)
_ = ax.set_title("Coefficients des modèles linéaires \nobtenus par validation croisée")

# %% [markdown]
#
# Nous pouvons remarquer que nos modèles ne souffrent pas d'instabilité
# numérique comme nous l'avons constaté lors de l'usage du modèle
# `LinearRegression`. Nous pouvons même observer la correspondance entre les
# coefficients des modèles `Ridge` et les coefficients utilisés pour générer
# les données originales. En effet, puisque que les variables prédictives
# originales sont répétées trois fois, les coefficients des modèles `Ridge`
# sont trois fois moins importantes.

# %%
true_coef

# %%
coef[true_coef.index].mean() * 3

# %% [markdown]
#
# Nous allons maintenant confirmer expérimentalement, l'effet du paramètre
# $\alpha$ sur les coefficients des modèles `Ridge`. Nous allons utiliser
# une valeur très faible et une valeur très élevée.

# %%
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

alphas = [1e-14, 1e5]
for ax, alpha in zip(axes, alphas):
    modele = Ridge(alpha=alpha)
    cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
    cv_results = cross_validate(
        modele, X, y, cv=cv, return_train_score=True, return_estimator=True
    )

    coef = pd.DataFrame(
        [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
        columns=nom_colonnes,
    )
    coef.plot.box(vert=False, whis=100, ax=ax)
    ax.set_title(r"Coefficients for $\alpha={}$".format(alpha))
fig.subplots_adjust(hspace=0.4)
_ = fig.suptitle(r"Effect of the parameter $\alpha$")

# %% [markdown]
#
# Nous pouvons donc confirmer nos premières intuitions. Plus $\alpha$ augmente,
# plus les coefficients des modèles `Ridge` sont plus petits. Au contraire,
# plus $\alpha$ diminue, le modèle se rapproche d'une simple régression
# linéaire, souffrant alors d'instabilité numérique.
#
# Maintenant, au lieu d'utiliser la norme L2 comme contrainte imposer sur les
# coefficients, nous pouvons utiliser une autre type de norme. Une possibilité
# est d'utiliser la norme L1 :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(X_i, \beta)
# \right)^2 + \alpha \|\beta\|_1
# $$
#
# Ce modèle est appelé **Lasso regression**. Nous pouvons obtenir les mêmes
# intuitions que précédemment : lorsque $\alpha$ est petit, le terme
# contraignant les coefficients est négligeable et donc nous avons une simple
# régression linéaire. Sinon, nous forcons la contraintes L1 sur les
# coefficients. La question est maintenant de comprendre quel est l'effet de
# minimizer la norme L1 sur les coefficients. Rien de tel que de répéter
# l'expérience précédente pour le modèle `Lasso`.

# %%
from sklearn.linear_model import Lasso

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

alphas = [1e-14, 1e4]
for ax, alpha in zip(axes, alphas):
    modele = Lasso(alpha=alpha)
    cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
    cv_results = cross_validate(
        modele, X, y, cv=cv, return_train_score=True, return_estimator=True
    )

    coef = pd.DataFrame(
        [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
        columns=nom_colonnes,
    )
    coef.plot.box(vert=False, whis=100, ax=ax)
    ax.set_title(r"Coefficients for $\alpha={}$".format(alpha))
fig.subplots_adjust(hspace=0.4)
_ = fig.suptitle(r"Effect of the parameter $\alpha$")

# %% [markdown]
#
# Nous pouvons constater que la minimiser la norme L1 forcera certains
# coefficients à devenir nuls. Si le paramètre $\alpha$ est trop grand, les
# coefficients seront tous nul. Nous pouvons constater que le problème que nous
# essayons de résoudre `Lasso` est performant car il permet de n'utiliser
# seulement qu'une des variables prédictives et n'utilise pas les variables
# colinéaires. En renvanche, il est intéressant que le choix de cette variable
# parmi les variables colinéaires peut-être arbitraire.

# %%
modele = Lasso(alpha=1, selection="random")
cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
cv_results = cross_validate(
    modele, X, y, cv=cv, return_train_score=True, return_estimator=True
)

coef = pd.DataFrame(
    [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
    columns=nom_colonnes,
)
ax = coef.plot.box(vert=False, whis=100)
_ = ax.set_title("Effet du paramètre selection")

# %% [markdown]
#
# Dans ce graphique, nous pouvons observer que la variation des valeurs des
# coefficients est réellement importante. Cela signifie que par exemple, la
# colonne 0, 6 ou 8 peuvent être sélectionnées arbitrairement. Pour une
# itération de validation croisée donnée, si la colonne 0 a été selectionnée,
# alors `Lasso` ne sélectionnera pas la colonne 6 ou 8. Et ceci peut changer
# d'itération à l'itération suivante. Utiliser le paramètre
# `selection="cyclic"` propose une sélection qui ne variera pas d'itération à
# l'itération suivante.
#
# Le dernier type de régularization que nous allons aborder est une combinaison
# entre `Ridge` et `Lasso`. La fonction de coût est alors définit comme suit :
#
# $$
# \mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N \left( y_i - f(X_i, \beta)
# \right)^2 + \alpha_1 \|\beta\|_1 + \alpha_2 \|\beta\|_2^2
# $$
#
# Ce modèle est appelé **Elastic Net regression**. Il permet alors de
# sélectionner un sous-ensemble des variables en utilisant la norme L1 comme
# dans `Lasso` et d'imposer une contrainte de type L2 sur les variables
# restantes. `scikit-learn` fournit un classe `ElasticNet` qui permet de
# construire un tel modèle. En revanche, il n'expose pas deux paramètres
# $\alpha_1$ et $\alpha_2$ mais un seul paramètre $\alpha$ commun aux deux type
# de contrainte L1 et L2 et un autre paramètre `l1_ratio` qui permet de qui
# permet de définir si nous favorison un modèle de type `Lasso` ou un modèle de
# type `Ridge`.

# %%
from sklearn.linear_model import ElasticNet

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

l1_ratios = [0.1, 0.9]
for ax, l1_ratio in zip(axes, l1_ratios):
    modele = ElasticNet(alpha=10, l1_ratio=l1_ratio)
    cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
    cv_results = cross_validate(
        modele, X, y, cv=cv, return_train_score=True, return_estimator=True
    )

    coef = pd.DataFrame(
        [modele_cv.coef_ for modele_cv in cv_results["estimator"]],
        columns=nom_colonnes,
    )
    coef.plot.box(vert=False, whis=100, ax=ax)
    ax.set_title(f"Coefficients for {l1_ratio=}")
fig.subplots_adjust(hspace=0.4)
_ = fig.suptitle("Effect of the parameter l1_ratio\n dans ElasticNet", y=1.05)

# %% [markdown]
#
# Nous pouvons constater que quand le paramètre `l1_ratio` est plus élevé,
# alors nour avons un modèle qui force les variables random (e.g. colonne 2 à
# 5) à être nulle. Lorsque le paramètre `l1_ratio` est plus faible, alors les
# variables random (e.g. colonne 2 à 5) sont plus non nuls ce qui
# correspondrait plus ò un modèle de type `Ridge`.

# %% [markdown]
#
# ### Importance du prétraitement
#
# ### De la régression à la classification
#
# ## Résolution de problèmes non-linéaires
#
# ## Quantifier l'incertitude des prédictions
