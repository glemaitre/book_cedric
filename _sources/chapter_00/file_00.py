# %% [markdown]
#
# # Introduction
#
# ## Motivations de l'apprentissage automatique
#
# Avant de rentrer dans le vive du sujet et découvrir les différents modèles
# d'apprentissage automatique disponibles dans `scikit-learn`, il advient de
# motiver l'usage de telles techniques.
#
# Nous pouvons d'ores et déjà définir ce que nous appelerons un modèle
# prédictif. Un modèle prédictif correspond à une fonction prenant des données
# en entrée est capable de renvoyé une prédiction en sortie. Mathématiquement,
# nous pouvons exprimer un tel modèle comme suit :
#
# $$
# \hat{y} = f(X) \,
# $$
#
# ou $X$ seront nos **données d'entrée**, $f()$ est notre **modèle prédictif**
# et $\hat{y}$ seront nos **prédictions**.
#
# Nous pouvons donner un exemple concret dès maintenant. Nous allons charger un
# jeu de données minimaliste.

# %%
import pandas as pd

donnees = pd.read_csv("../datasets/penguins_regression.csv")
donnees.head()

# %% [markdown]
#
# Nous venons de charger en mémoire des données contenant des informations
# de pingouins. Ce jeu de données contient deux informations : pour chaque
# pingouin, nous connaissons la longeur de son aileron ainsi que sa masse
# corporelle.
#
# Nous pouvons formuler le problème de prédiction suivant : à partir de ces
# données, nous souhaitons apprendre une fonction $f()$ qui prend en entrée
# la longueur d'aileron d'un pingouin et qui renvoie sa masse corporelle.
#
# Nous pouvons dès à présent visualiser ces données.

# %%
import seaborn as sns

sns.set_context("poster")

ax = donnees.plot.scatter(x=donnees.columns[0], y=donnees.columns[1])
_ = ax.set_title("Peut-on prédire la masse corporelle \nd'un pingouin?")

# %% [markdown]
#
# En oberservant nos données, nous pouvons constater que nous avons un jeu de
# données contenant à la fois l'entrée et la sortie de notre fonction $f()$.
# Typiquement, ceci est le cas lorsque nous sommes dans un contexte
# d'**apprentissage supervisé**.
#
# L'idée sera de trouver la fonction $f()$ optimale à partir de ces données
# d'entrée et sortie. Par la suite, nous utiliserons cette fonction pour
# prédire la sortie à partir d'une nouvelle donnée d'entrée.
#
# Dans `scikit-learn`, les données d'entrée et de sortie sont séparées en deux
# tableaux distincts, dénotés `X` et `y` dans la documentation.

# %%
X = donnees[["Longueur Aileron (mm)"]]
y = donnees["Masse Corporelle (g)"]

# %% [markdown]
#
# Nous allons nous attarder sur la structure de ces données.

# %%
X.head()

# %% [markdown]
#
# `X` est représenté par une **matrice** de données en deux dimensions. Chaque
# ligne correspond à un **échantillon** (i.e. un pingouin) et chaque colonne
# correspond à une **variable** (i.e. une mesure physique). Dans notre jeu de
# données minimaliste, nous avons une seule variable d'entrée, `Longueur
# Aileron (mm)`.

# %%
y.head()

# %% [markdown]
#
# `y` est représenté par un **vecteur** de données (i.e. une seul dimension).
# Nous avons autant d'éléments que de lignes dans `X`. Cette variable est
# communemment appellée **variable cible**.
#
# Il important de noter que `y` est une variable continue : elle peut prendre
# n'importe quel valeur entre moins l'infini et plus l'infini ou quasiment
# (nous somme limités par la physique puisque le masse corporelle d'un pingouin
# ne peut-être négative et trop grande). Lorsque `y` est une **variable
# continue** nous appelons ce problème de prédiction, un problème de
# **regression**.
#
# Si `y` est une variable **discrète**, comme par exemple l'espèce de pinguoin,
# nous appelons ce problème, un problème de **classification**.
#
# Donc à partir de ces données, nous aimerions apprendre une fonction de
# prédiction `f()`. La définition de telles fonctions sont le résultat de
# recherche fondamentale dans le domaine de l'apprentissage automatique. Nous
# allons présenter ces différentes fonctions dans la suite de ce livre.
#
# Mais afin d'illustrer le principe d'une fonction de prédiction, nous
# pourrions venir avec notre propre fonction de prédiction, basée sur une
# intuition. En regardant le graphique représentant notre variable d'entrée et
# notre variable de sortie, nous pourrions calculer une relation moyenne entre
# la longueur d'aileron et la masse corporelle.

# %%
import numpy as np

relation_X_y = np.mean(X["Longueur Aileron (mm)"] / y)
relation_X_y

# %% [markdown]
#
# Maintenant nous pouvons utiliser cette relation pour prédire la masse :
# nous sommes entrain de d'utiliser le fameux "produit en croix". Donc,
# essayons de prédire la masse du premier pingouin dans notre dataset.

# %%
masse_premier_pinguoin = X.iloc[0, 0] / relation_X_y
masse_premier_pinguoin

# %% [markdown]
#
# Nous pouvons même estimer la différence avec sa masse réelle.

# %%
print(
    f"Masse réelle: {y[0]} grammes\n"
    f"Masse prédite: {masse_premier_pinguoin:.2f} grammes\n"
    f"Différence: {y[0] - masse_premier_pinguoin:.2f} grammes"
)

# %% [markdown]
#
# Pour pousser notre exemple plus proche de notre définition mathématique,
# nous pouvons utiliser une fonction Python pour apprendre la relation et
# une autre fonction Python pour prédire la masse.


# %%
def apprendre_relation(X, y):
    """Apprendre une relation moyenne entre la longueur d'aileron et la masse
    corporelle des pinguoins."""
    return np.mean(X["Longueur Aileron (mm)"] / y)


def f(X, relation):
    """Function de predisant la masse corporelle d'un pingouin à partir de
    la longueur de son aileron."""
    return (X["Longueur Aileron (mm)"] / relation).rename("Masse Corporelle (g)")


# %%
relation = apprendre_relation(X, y)
predictions = f(X, relation)
predictions

# %% [markdown]
#
# Nous pouvons même calculer l'erreur (absolue) de nos prédictions.

# %%
erreur_absolue_moyenne = np.mean(np.abs(predictions - y))
print(f"Erreur absolue moyenne: {erreur_absolue_moyenne:.2f} grammes")

# %% [markdown]
#
# Notre stratégie d'apprentissage nous permet d'obtenir un modèle prédictif
# qui commet en moyenne une erreur de 467 grammes sur le même jeu de données.
# Nous pouvons également représenter graphiquement la relation que nous avons
# appris.

# %%
ax = donnees.plot.scatter(x=donnees.columns[0], y=donnees.columns[1])
ax.plot(
    X["Longueur Aileron (mm)"],
    predictions,
    color="black",
    linewidth=4,
    label="Notre fonction",
)
ax.legend()
_ = ax.set_title("Notre premier modèle prédictif")

# %% [markdown]
#
# Il est a noté que nous ne devrions pas apprendre et evaluer notre fonction de
# prédiction sur les mêmes données. Le score obtenu est potentiellement trop
# optimiste. Nous allons revenir sur ce point à la fin de cette section où nous
# présenterons les outils et la manière d'évaluer correctement un modèle
# prédictif.
#
# ## Introduction à l'interface de programmation de `scikit-learn`
#
# Dans cette section, nous allons présenter succintement l'interface de
# programmation de `scikit-learn`. Nous utiliserons cette interface dans les
# sections et chapitres suivants lorsque nous présenterons les différents types
# de modèles prédictifs.
#
# Afin de présenter cette interface, nous allons créer nous même un modèle
# prédictif qui pourrait être utilisé en conjonction avec tous les outils de
# `scikit-learn`. Nous allons également argumenté les choix d'interface
# réalisés par les développeurs de `scikit-learn` historiquement.
#
# Dans la précédente section, nous avons vu constaté que l'apprentissage
# automatique était consititué de deux étapes : une **étape d'apprentissage**
# pour obtenir la fonction de prédiction optimale et une **étape de
# prédiction** ou nous avons réutiliser la relation appris pour prédire la
# masse corporelle.
#
# Il advient que nous pourrions utiliser une classe Python pour stocker à
# l'intérieur de cette instance d'objet les informations utilisées pour la
# prédiction. Nous aurions seulement besoin donc d'exposer une méthode pour
# effectuer l'apprentissage et une méthode pour prédire.
#
# En résumé, nous pouvons créer la classe suivante :


# %%
from sklearn.base import BaseEstimator


class ModelePredictif(BaseEstimator):
    def fit(self, X, y):
        self.coef_ = np.mean(X["Longueur Aileron (mm)"] / y)
        return self

    def predict(self, X):
        return (X["Longueur Aileron (mm)"] / relation).rename("Masse Corporelle (g)")


# %% [markdown]
#
# Dans `scikit-learn`, un modèle prédictif (également appelé estimateur)
# possède deux méthodes : `fit` et `predict`. La première est en charge
# d'apprendre les éléments nécessaires à la prédiction. De plus cette méthode
# retourne `self` pour permettre de chainer les appels. La deuxième méthode est
# en charge de prédire. Nous pouvons mettre en place notre class de la façon
# suivante :

# %%
modele = ModelePredictif()
modele.fit(X, y)
predictions = modele.predict(X)
erreur_absolue_moyenne = np.mean(np.abs(predictions - y))
print(f"Erreur absolue moyenne: {erreur_absolue_moyenne:.2f} grammes")

# %% [markdown]
#
# En plus des méthodes `fit` et `predict`, `scikit-learn` expose une méthode
# `score` qui permet de calculer le score de prédiction.


# %%
class ModelePredictif(BaseEstimator):
    def fit(self, X, y):
        self.coef_ = np.mean(X["Longueur Aileron (mm)"] / y)
        return self

    def predict(self, X):
        return (X["Longueur Aileron (mm)"] / relation).rename("Masse Corporelle (g)")

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(np.abs(predictions - y))


# %% [markdown]
#
# Nous pourrions donc simplifier notre code précédent de la manière suivante :

# %%
print(f"Erreur absolue moyenne: {ModelePredictif().fit(X, y).score(X, y):.2f} grammes")

# %% [markdown]
#
# ## Evaluation de notre modèle prédictif
#
# Dans la section précédente, nous avons calculé l'erreur absolue moyenne sur
# le même jeu de données utilisé pendant l'apprentissage.
#
# Cependant cette approche est réellement problématique. Nous aurions pu créer
# un modèle prédictif qui ne prend pas en compte les données d'apprentissage et
# qui mémorise tous les échantillons d'apprentissage. Nous aurions alors donc
# obtenu une erreur de prédiction de 0 grammes. En revance, en utilisant le
# même modèle sur un nouveau dataset pour seulement effectuer de la prédiction,
# nous aurions alors obtenu une erreur plus importante.
#
# C'est pour cela qu'il est nécessaire d'évaluer un modèle prédictif sur des
# données qui ne sont pas dans le jeu d'apprentissage. L'erreur obtenu sur le
# jeu d'apprentissage est appelé l'**erreur empirique** alors que l'erreur
# obtenu sur le jeu de test est appelé l'**erreur de généralisation**. En
# apprentissage automatique, les méthodes essayent en général de minimiser
# l'erreur empirique en espérant que l'erreur de généralisation soit minimale
# également.
#
# Nous pouvons utiliser notre jeu de données original et le séparer en deux
# jeux de données afin de calculer les deux types d'erreurs. `scikit-learn`
# fournit une fonction `train_test_split` qui permet de séparer un jeu de
# données.

# %%
from sklearn.model_selection import train_test_split

X_apprentissage, X_test, y_apprentissage, y_test = train_test_split(
    X,
    y,
    random_state=0,
    test_size=0.2,
    shuffle=True,
)

# %% [markdown]
#
# La sélection de ces jeux de données est faite de manière aléatoire. Le
# paramètre `random_state` permet d'obtenir une séparation déterministique même
# si un procédure aléatoire est utilisée. De plus, le paramètre `test_size`
# permet de définir la proportion de données qui seront utilisées pour le jeu
# de test.

# %%
modele = ModelePredictif().fit(X_apprentissage, y_apprentissage)

# %%
print(f"Erreur empirique: {modele.score(X_apprentissage, y_apprentissage):.2f} grammes")
print(f"Erreur de généralisation: {modele.score(X_test, y_test):.2f} grammes")

# %% [markdown]
#
# Nous observons que notre modèle performe moins bien sur le jeu de test que
# sur le jeu d'apprentissage.
#
# La comparaison des erreurs empérique et de généralisation permette de
# connaître de savoir si un modèle sous-apprend, généralise ou sur-apprend. Le
# sur-apprentissage est caractérisé par une erreur de géralisation plus élevée
# que l'erreur empirique. La plage de générasation est définie par lorsque la
# différence entre l'erreur de généralisation et l'erreur empirique est assez
# minimale. Un modèle sous-apprentissage est caractérisé par une erreur de
# généralisation élevée mais également une erreur empirique élevée.
#
# Bien que nous ayons maintenant une idée concernant la capacité de notre
# modèle à prédire sur un set indépendant de celui d'apprentissage, nous ne
# pouvons pas réellement savoir si la différence entre les erreurs empirique et
# de généralisation est suffisante pour déterminer si un modèle est en
# sous-apprentissage ou en sur-apprentissage ou généralise. En effet, puisque
# la procédure de séparation des jeux de données est aléatoire, nous devrions
# réaliser plusieurs essais pour obtenir une distribution des erreurs et
# evaluer si les différences sont suffisantes.
#
# Ces essais répétés sont dénommés **cross-validation**. `scikit-learn` fournit
# une fonction `cross_validate` permettant de réaliser ces essais. Il existe
# plusieurs méthodes de cross-validation. Nous allons utiliser la méthode
# **k-fold** qui permet de séparer un jeu de données en $k$ sous-ensembles. A
# chaque itération de la cross-validation, un des $k$ sous-ensembles est
# utilisé pour évaluer le modèle alors que les autres sous-ensembles sont
# utilisés pour l'apprentissage.

# %%
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_resultats = cross_validate(
    modele, X, y, cv=cv, scoring="neg_mean_absolute_error", return_train_score=True
)
cv_resultats = pd.DataFrame(cv_resultats)
cv_resultats[["train_error", "test_error"]] = -cv_resultats[
    ["train_score", "test_score"]
]

# %%
cv_resultats[["train_error", "test_error"]]

# %% [markdown]
#
# Cross-validation nous permet donc d'obtenir plusieurs estimations de nos
# erreurs. Nous pouvons même répéter l'opération plusieurs fois pour obtenir
# des distributions d'erreurs que nous pourrons visualiser.

# %%
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=5, n_repeats=30, random_state=0)
cv_resultats = cross_validate(
    modele, X, y, cv=cv, scoring="neg_mean_absolute_error", return_train_score=True
)
cv_resultats = pd.DataFrame(cv_resultats)
cv_resultats[["train_error", "test_error"]] = -cv_resultats[
    ["train_score", "test_score"]
]

# %%
ax = cv_resultats[["train_error", "test_error"]].plot.hist(bins=20, alpha=0.5)
ax.set_xlabel("Erreur moyenne absolue (grammes)")
_ = ax.set_title(
    f"Distribution des erreurs\nStd. dev. variable cible: {np.std(y):.2f} (g)"
)

# %% [markdown]
#
# Il est toujours bon de mettre en perspective l'erreur moyenne obtenue avec
# la variable cible. En effet, notre modèle commet une erreur de 450 grammes
# sur notre données alors que la standard deviation de la variable cible est de
# 850 grammes. Notre modèle n'est donc pas un très bon modèle.
#
# Quand on compare les distributions, nous regardons si ces distributions se
# chevauchent. Si les distributions se chevauchent et que leur moyenne et leur
# standard déviation sont proches, alors le modèle est capable de généraliser.
# Une différence entre les distributions nous permet de conclure sur le sous-
# ou sur-apprentissage. En revanche, quand les standard déviations sont trop
# larges, ceci est souvent lié à un manque de données et il sera difficile de
# tirer des conclusions.
