# %%
import pandas as pd

bike_sharing = pd.read_csv("datasets/bike_sharing/hour.csv")

# %%
bike_sharing.head()

# %%
bike_sharing.info()

# %%
bike_sharing = pd.read_csv(
    "datasets/bike_sharing/hour.csv", parse_dates=["dteday"]
)

# %%
bike_sharing.info()

# %%
features = {
    "season": "saison",
    "yr": "annee",
    "mnth": "mois",
    "hr": "heure",
    "holiday": "vacances",
    "weekday": "jour de la semaine",
    "weathersit": "meteo",
    "temp": "temperature",
    "atemp": "temperature ressentie",
    "hum": "humidite",
    "windspeed": "vitesse du vent",
}
target = {"cnt": "nombre de locations"}

# %%
bike_sharing = bike_sharing[
    list(features.keys()) + list(target.keys())
]
# %%
bike_sharing = bike_sharing.rename(columns={**features, **target})

# %%
bike_sharing.head()

# %%
bike_sharing.describe()

# %%
import seaborn as sns
sns.set_context("talk")

# %%
_ = bike_sharing.hist(bins=20, grid=False, edgecolor="black", figsize=(15, 20))

# %%
import numpy as np

rng = np.random.RandomState(0)
indices = rng.choice(
    np.arange(bike_sharing.shape[0]), size=200, replace=False,
)

# %%
subset = bike_sharing.iloc[indices]
# Quantize the target and keep the midpoint for each interval
subset["nombre de locations"] = pd.qcut(
    subset["nombre de locations"], 6, retbins=False,
)
subset["nombre de locations"] = subset["nombre de locations"].apply(
    lambda x: x.mid
)

# %%
_ = sns.pairplot(data=subset, hue="nombre de locations", palette="viridis")

# %%
