import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as pl

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import log_loss as LOGLOSS

#   
# BASIC SHAP MODEL
# VARS CONSIDERED: (goldearned, goldspent, kills, duration) per min
#

shap.initjs()

# read in the data
prefix = "C:/Users/Alex/Desktop/TUFTS/SEMESTER 3/Probabilistic Sys Anal/FinalProj/archive/"
matches = pd.read_csv(prefix+"matches.csv")
participants = pd.read_csv(prefix+"participants.csv")
stats1 = pd.read_csv(prefix+"stats1.csv", low_memory=False)
stats2 = pd.read_csv(prefix+"stats2.csv", low_memory=False)
stats = pd.concat([stats1,stats2])

# merge into a single DataFrame
a = pd.merge(participants, matches, left_on="matchid", right_on="id")
allstats_orig = pd.merge(a, stats, left_on="matchid", right_on="id")
allstats = allstats_orig.copy()

# drop games that lasted less than 10 minutes
allstats = allstats.loc[allstats["duration"] >= 10*60,:]

# Convert string-based categories to numeric values
cat_cols = ["role", "position", "version", "platformid"]
for c in cat_cols:
    allstats[c] = allstats[c].astype('category')
    allstats[c] = allstats[c].cat.codes
allstats["wardsbought"] = allstats["wardsbought"].astype(np.int32)

# List of features to exclude during model training
# Must ALWAYS exclude win to prevent disaster
rate_features_rm = [
    "win", "deaths", "assists", "killingsprees", "doublekills",
    "triplekills", "quadrakills", "pentakills", "legendarykills",
    "totdmgdealt", "magicdmgdealt", "physicaldmgdealt", "truedmgdealt",
    "totdmgtochamp", "magicdmgtochamp", "physdmgtochamp", "truedmgtochamp",
    "totheal", "totunitshealed", "dmgtoobj", "timecc", "totdmgtaken",
    "magicdmgtaken" , "physdmgtaken", "truedmgtaken",
    "totminionskilled", "neutralminionskilled", "ownjunglekills",
    "enemyjunglekills", "totcctimedealt", "pinksbought", "wardsbought",
    "wardsplaced", "wardskilled", "turretkills", "inhibkills", "champlvl", 
    "firstblood", "visionscore", "dmgtoturrets", "dmgselfmit", "largestcrit", 
    "longesttimespentliving", "largestmultikill", "largestkillingspree", "role", 
    "platformid", "ss2", "seasonid", "queueid", "version", "trinket",
    "item3", "item4", "creation", "gameid", "item1", "item2", "item5", "item6", 
    "id_x", "matchid", "player", "championid", "ss1", "position", "id_y", "id"
]

X = allstats.drop(rate_features_rm, axis=1)
for i in X:
    print(X[i])
y = allstats["win"]

for feature_name in X:
    X[feature_name] /= X["duration"] / 60 # per minute rate

#X["goldearned"] /= X["duration"] / 60 # per minute rate
#X["goldspent"] /= X["duration"] / 60 # per minute rate


# convert to fraction of game
#X["longesttimespentliving"] /= X["duration"]

# define friendly names for the features
full_names = {
    "goldearned": "Gold earned per min",
    "goldspent": "Gold spent per min",
    "kills": "Kills per min",
    "win": "Win or Loss"
}
feature_names = [full_names.get(n, n) for n in X.columns]
X.columns = feature_names

# create train/validation split
Xt, Xv, yt, yv = train_test_split(X,y, test_size=0.2, random_state=10)
dt = xgb.DMatrix(Xt, label=yt.values)
dv = xgb.DMatrix(Xv, label=yv.values)

params = {
    "eta": 0.5,
    "max_depth": 4,
    "objective": "binary:logistic",
    "silent": 1,
    "base_score": np.mean(yt),
    "eval_metric": "logloss"
}
model = xgb.train(params, dt, 300, [(dt, "train"),(dv, "valid")], early_stopping_rounds=5, verbose_eval=25)

# compute the SHAP values for every prediction in the validation dataset
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xv)

shap.force_plot(explainer.expected_value, shap_values[0,:], Xv.iloc[0,:])

shap.summary_plot(shap_values, Xv)

#xs = np.linspace(-4,4,100)
#pl.xlabel("Log odds of winning")
#pl.ylabel("Probability of winning")
#pl.title("How changes in log odds convert to probability of winning")
#pl.plot(xs, 1/(1+np.exp(-xs)))
#pl.show()

# sort the features indexes by their importance in the model
# (sum of SHAP value magnitudes over the validation dataset)
shap.dependence_plot("Gold earned per min", shap_values, Xv, interaction_index="Gold spent per min")
shap.dependence_plot("Kills per min", shap_values, Xv, interaction_index="Gold earned per min")


# --------------------------------
# Accuracy / Error measurement
# Assuming 'y_v' is  ground truth labels
# --------------------------------

y_test_pred = model.predict(dv)

mse = MSE(y_test_pred, yv)
print("MSE : % f" %(mse)) 

logloss = LOGLOSS(yv,y_test_pred)
print("LOGLOSS : % f" %(logloss))