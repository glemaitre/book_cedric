# %%
# import pandas as pd
# import seaborn as sns
# sns.set_context("talk")

# bike_sharing = pd.read_csv(
#     "datasets/bike_sharing/hour.csv", parse_dates=["dteday"]
# )

# features = {
#     "season": "saison",
#     "yr": "annee",
#     "mnth": "mois",
#     "hr": "heure",
#     "holiday": "vacances",
#     "weekday": "jour de la semaine",
#     "weathersit": "meteo",
#     "temp": "temperature",
#     "atemp": "temperature ressentie",
#     "hum": "humidite",
#     "windspeed": "vitesse du vent",
# }
# target = {"cnt": "nombre de locations"}

# bike_sharing = bike_sharing[
#     list(features.keys()) + list(target.keys())
# ]
# bike_sharing = bike_sharing.rename(columns={**features, **target})

# # %%
# bike_sharing.info()

# # %%
# data = bike_sharing[features.values()].select_dtypes("floating")
# target = bike_sharing[target.values()]

# # %%
# import numpy as np

# rng = np.random.RandomState(0)
# indices = rng.choice(np.arange(data.shape[0]), size=1_000)
# subset = pd.concat([data, target], axis=1).iloc[indices].copy()

# # %%
# _ = subset.hist(bins=20, grid=False, edgecolor="black", figsize=(10, 10))

# # %%
# _ = sns.pairplot(subset)

# # %%
# from sklearn.decomposition import PCA

# pca = PCA(n_components=None)
# pca.fit(data)

# # %%
# pca.explained_variance_ratio_

# # %%
# projection = pd.DataFrame(
#     pca.components_ * np.sqrt(pca.explained_variance_),
#     columns=[f"PC #{i+1}" for i in range(pca.explained_variance_.size)],
#     index=data.columns,
# )

# # %%
# _ = sns.heatmap(projection)

# # %%
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import RidgeCV

# alphas = np.logspace(-3, 3, num=50)
# model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

# # %%
# from sklearn.model_selection import cross_validate

# cv_results = cross_validate(
#     model, data, target, scoring="neg_mean_absolute_error", n_jobs=-1,
# )
# error = -cv_results["test_score"]

# # %%
# print(f"{error}\n{error.mean():.3f} +/- {error.std():.3f}")

# # %%
# model = make_pipeline(
#     StandardScaler(), PCA(n_components=2), RidgeCV(alphas=alphas)
# )

# # %%
# cv_results = cross_validate(
#     model, data, target, scoring="neg_mean_absolute_error", n_jobs=-1,
# )
# error = -cv_results["test_score"]
# print(f"{error}\n{error.mean():.3f} +/- {error.std():.3f}")

# # %%
# model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
# for feature_name in data.columns:
#     cv_results = cross_validate(
#         model, data[[feature_name]], target,
#         scoring="neg_mean_absolute_error", n_jobs=-1
#     )
#     error = -cv_results["test_score"]
#     print(f"Keeping only {feature_name} column")
#     print(f"{error}\n{error.mean():.3f} +/- {error.std():.3f}")

# # %%
