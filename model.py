# import libraries
import pandas as pd
import io, time, json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
from scipy import stats
import pandas_profiling as pp
import warnings
warnings.filterwarnings('ignore')
from scipy.special import boxcox1p
import re
import folium
from folium.plugins import HeatMap
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.cross_validation import KFold
import matplotlib; matplotlib.use('TkAgg')
import lightgbm as lgb


trulia_file = r"./scrape_data/trulia_information.csv" 
yelp_ws_file = r"./scrape_data/yelp_ws_information.csv"

trulia = pd.read_csv(trulia_file, thousands=",")
yelp_ws = pd.read_csv(yelp_ws_file)
df = trulia.merge(yelp_ws, on = ["address"], how = 'left')
df.to_csv(r"./scrape_data/data.csv",index = False)


df = pd.read_csv(r"./scrape_data/data.csv", thousands=",")
pd.isna(df).sum() / len(df) * 100



# drop rows with missing price and built year
df = df[(pd.notna(df.price))]
df = df[(pd.notna(df.built_year))]
df = df[df['area'] > 0]


# This function remove unit in lot size column and change all unit to squarefeet.
def conver_lotsize(df):
    res = []
    for i in df["lot_size"]:
        if isinstance(i,float):
            res.append(i)
            continue
        ii = i.split(" ")
        if  ii[-1] == 'acres' or ii[-1] == 'acre':

            t = ii[0]
            r = "".join(t.split(","))
            res.append(43560 * float(r))
            continue
        if ii[-1] == 'sqft':
            t = ii[0]
            r = "".join(t.split(","))
            res.append(float(r))
        else:
            res.append(i)
    df["lot_size"] = res
    return df

df = conver_lotsize(df)

drop_list = ["address", "link", "sold information","heating_fuel","architecture_type", "exterior", "bike", "number_of_rooms", "parking_spaces", "roof"]
df = df.drop(drop_list, axis = 1)


df["cooling_system"].fillna("None", inplace = True)
df["heating_system"].fillna("None", inplace = True)
df["parking"].fillna("None", inplace = True)
df["stories"].fillna("Unknown", inplace = True)


lotsize_mean = df[df["lot_size"] != -1]["lot_size"].median()
df["lot_size"].fillna(lotsize_mean, inplace = True)
df["walk"].fillna(df.walk.median(), inplace = True)
df["arts_count"].fillna(0, inplace = True)
df["grocery_count"].fillna(0, inplace = True)


df = df[np.abs(df.price - df.price.mean()) <= (3 * df.price.std())]


datatype = {
            "dishwasher": np.str,
            "microwave": np.str, 
            "washer":np.str, 
            "dryer":np.str, 
            "refrigerator":np.str, 
            "number_beds": np.int64,
            "number_baths": np.int64,
            "built_year": np.int64,
            "price": np.int64,
            "postcode": np.str
            }
 
df = df.astype(datatype)
df.reset_index(inplace=True, drop=True)


def getdf(path):
    train_df = pd.read_csv(path)
    train_df = train_df.replace('None', np.nan)
    train_df = train_df.replace('Unknown', np.nan)
    return train_df
    

# display plot of numerical features that has high skewness
def skewIndex(df):
    features = df[df.columns].skew(axis = 0, skipna = True)
    high_skewness = features[features > 0.5]
    skew_index = high_skewness.index
    return skew_index


def getDescription(df):
    skew_index = skewIndex(df)
    for c in skew_index:
        print("\n")
        print("-------------------------")
        print(c, "statics:")
        print(df[c].describe()) 
    sn.distplot(train_df["price"])
    print("Skewness: %f" % train_df["price"].skew())
    print("Kurtosis: %f" % train_df["price"].kurt())
    

train_df = getdf("data_eda.csv")
getDescription(train_df)


def plot(df):
    skew_index = skewIndex(df)
    train_df[skew_index].hist(figsize=(15,15))
    for i in range(len(skew_index)):
        #scatter plot
        
        var = skew_index[i]
        
        if var == "price":
            continue 
        data = pd.concat([train_df['price'], train_df[var]], axis=1)
        data.plot.scatter(x=var, y='price', ylim=(0,800000))
        
plot(train_df)



corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]

cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


color = sns.color_palette()
cols = [col for col in df.columns if col not in ['price'] if df[col].dtype=='float64' or df[col].dtype=='int64']

labels = []
values = []
for col in cols:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df.price.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(10,10))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()



plt.figure(figsize=(8,6))
sns.boxplot(x="number_baths", y="price", data=df)
plt.ylabel('price', fontsize=11)
plt.xlabel('Bathroom numbert', fontsize=11)
plt.xticks(rotation='vertical')
plt.title("price VS bathroom count?", fontsize=14)
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(x="number_beds", y="price", data=df)
plt.ylabel('price', fontsize=11)
plt.xlabel('Bedroom numbert', fontsize=11)
plt.xticks(rotation='vertical')
plt.title("price VS bedroom count?", fontsize=14)
plt.show()



col = "area"
ulimit = np.percentile(df[col].values, 99.5)
llimit = np.percentile(df[col].values, 0.5)
df[col].ix[df[col]>ulimit] = ulimit
df[col].ix[df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=df['area'].values, y=df['price'].values, size=10, color='g')
plt.ylabel('price', fontsize=11)
plt.xlabel('area', fontsize=11)
plt.title("area VS price", fontsize=14)
plt.show()



var = 'built_year'
data = pd.concat([df['price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="price", data=data)
plt.xticks(rotation=90);



# ## Data Preprocessing

# #### Price Distribution Analysis
# reference: https://brianmusisi.com/design/Predicting+House+Prices-2.html

fig = plt.figure(figsize=(8, 6))
sns.distplot(df['price'], fit=norm)
plt.ylabel('frequency')
plt.title('price distribution')

fig = plt.figure(figsize=(8, 6))
stats.probplot(df['price'], plot = plt)
plt.show()

print("Skewness: " + str(df["price"].skew()))
print("Kurtosis: " + str(df["price"].kurt()))


df['price_t'] = np.log1p(df["price"] + 88000)
fig = plt.figure(figsize=(8, 6))
sns.distplot(df['price_t'],fit=norm)
plt.ylabel('frequency')
plt.title('price distribution')

fig = plt.figure(figsize=(8, 6))
stats.probplot(df["price_t"], plot=plt)
plt.show()

print("Skewness: " + str(df["price_t"].skew()))
print("Kurtosis: " + str(df["price_t"].kurt()))


feature_examination = ["lot_size", "restaurant_count", 'area', 'arts_count', 'walk',
                       'grocery_count', 'number_baths', 'number_beds',
                       'restaurant_rating', 'restaurant_price', 'transit' ]
feature_skewness = {}
for i in feature_examination:
    feature_skewness[i] = skew(df[i])
s = pd.DataFrame.from_dict(feature_skewness, orient='index')


s_1 = s[(s[0] > 1 )]
res = list(s_1.index)
new = []
df = df[df['area'] > 0]
for i in res:
    new_column = i + "_t"
    new.append(new_column)
    df[new_column] = boxcox1p(df[i], 0.1)
s_n1 = s[(s[0] < -1 )]
res_n = list(s_n1.index)
for i in res_n:
    new_column = i + "_t"
    new.append(new_column)
    df[new_column] = boxcox1p(df[i], 3)



feature_skewness_new = {}
for i in new:
    feature_skewness_new[i] = skew(df[i])
s_new = pd.DataFrame.from_dict(feature_skewness_new, orient='index')
s_new


# The `Box Cox Transformation` makes the features more normalized.


drop_list = ["price","lot_size", "restaurant_count", "area","arts_count",
             "grocery_count", "number_baths","restaurant_rating",
             "restaurant_price"]
df_model = df.drop(drop_list, axis = 1)
df_model



from sklearn.preprocessing import LabelEncoder
numerics = ['int64', 'float64']
df_num = df_model.select_dtypes(include = numerics)
df_cat = df_model.select_dtypes(include = "object")

df_process = pd.DataFrame()

# For each categorical column
# We fit a label encoder, transform our column and 
# add it to our new dataframe
label_encoders = {}
for col in df_cat.columns:
    print("Encoding " + col)
    new_label = LabelEncoder()
    df_process[col] = new_label.fit_transform(df_model[col])
    label_encoders[col] = new_label

df_model_label = pd.concat([df_num, df_process], axis = 1)
df_model_label.info()


# #### Training and Testing Dataset
# We shuffled the dataset when we did tehe project, but we do not want it to be ran again,
# so we removed code away.

train, test = train_test_split(df_model_label, test_size=0.25)
train.reset_index(inplace = True, drop = True)
test.reset_index(inplace = True, drop = True)
train.head()


# ## Predictive Model


# code modified from https://www.kaggle.com/flennerhag/ml-ensemble-scikit-learn-style-ensemble-learning#4.-Ensemble-learning
plt.figure(figsize=(23,23))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,
            square=True, cmap=plt.cm.RdBu, linecolor='black', annot=True)



# Random Forest Regressor
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_depth': 7,
    'min_samples_leaf': 2,
    'max_features' : 'auto'
}

# Extra Trees Regressor
et_params = {
    'n_jobs': -1,
    'n_estimators':1000,
    'max_depth': 7,
    'min_samples_leaf': 2,
    'max_features' : 'auto'
}

# AdaBoost Regressor
ada_params = {
    'n_estimators': 1000,
    'learning_rate' : 0.01
}

# Gradient Boost Regressor
gb_params = {
    'n_estimators': 1000,
    'subsample': 0.9,
    'max_depth': 7,
    'min_samples_leaf': 2
}



# source:https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python#Ensembling-&-Stacking-models
class SklearnHelper(object):
    def __init__(self, model, seed=0, params=None):
        params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self,x,y):
        return self.model.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.model.fit(x,y).feature_importances_)



train_y = train['price_t']
train_x = train.drop(['price_t'], axis=1)
test_y = test['price_t']
test_x = test.drop(['price_t'], axis=1)
train_x = np.array(train_x)
test_x = np.array(test_x)


num_train = len(train)
num_test = len(test)

rf = SklearnHelper(model=RandomForestRegressor, seed=0, params = rf_params)
et = SklearnHelper(model=AdaBoostRegressor, seed=0, params = ada_params)
ada = SklearnHelper(model=GradientBoostingRegressor, seed=0, params = gb_params)
gb = SklearnHelper(model=ExtraTreesRegressor, seed=0, params = et_params)



def k_train(model, x_train, y_train, x_test):
    train = np.zeros((num_train,))
    test = np.zeros((num_test,))
    test_k = np.empty((4, num_test))
    
    kf = KFold(num_train, n_folds= 4, random_state=0)
    for i, (train_index, test_index) in enumerate(kf):

        x_t = x_train[train_index]
        y_t = y_train[train_index]

        x_te = x_train[test_index]

        model.train(x_t, y_t)

        train[test_index] = model.predict(x_te)
        test_k[i, :] = model.predict(x_test)

    test[:] = test_k.mean(axis=0)
    return train.reshape(-1, 1), test.reshape(-1, 1)


# Create our OOF train and test predictions. These base results will be used as new features
et_train, et_test = k_train(et, train_x, train_y, test_x) # Extra Trees
print("finish extra trees")
rf_train, rf_test = k_train(rf,train_x, train_y, test_x) # Random Forest
print("finish random forest")
ada_train, ada_test = k_train(ada, train_x, train_y, test_x) # AdaBoost 
print("finish adaboost")
gb_train, gb_test = k_train(gb,train_x, train_y, test_x) # Gradient Boost
print("finish gradient boost")



rf_feature = rf.feature_importances(train_x,train_y)
et_feature = et.feature_importances(train_x, train_y)
ada_feature = ada.feature_importances(train_x, train_y)
gb_feature = gb.feature_importances(train_x,train_y)
print(rf_feature)



rf_feature = [4.52158674e-03,7.64134556e-02,2.61640801e-02,8.15664918e-02,
              2.11066467e-02,1.03792748e-02,3.79408396e-02,1.79405198e-02,
              4.36974195e-01,4.17593903e-02,1.94860800e-02,1.23580171e-02,
              7.96079350e-03,5.84109567e-02,2.40779393e-03,1.36511318e-03,
              3.56570501e-04,2.98341263e-04,5.65451478e-04,5.95093209e-03,
              4.88584809e-03,9.03332379e-02,2.76553223e-03,1.57968050e-03,
              3.65091711e-02]
et_feature = [9.62476564e-04,1.01007865e-01,8.07794820e-03,4.54525396e-02,
              3.60118968e-03,1.63143924e-03,3.10994579e-02,2.30151619e-02,
              4.12820815e-01,3.94423810e-02,1.69292880e-02,4.65838034e-02,
              2.80368272e-03,6.30150661e-02,2.76426807e-02,9.83603976e-03,
              7.22998419e-03,5.95944597e-03,8.95066691e-05,1.04661769e-02,
              1.87307475e-03,1.17996939e-01,1.80878903e-03,2.15503555e-03,
              1.84992125e-02]
ada_feature = [0.01424421,0.0665479,0.12907858,0.13272909,0.0645828,
               0.05124307,0.09418432,0.05147204,0.12862256,0.02011813,
               0.01717681,0.01121615,0.05395952,0.04655981,0.00474344,
               0.00329752,0.0024825,0.00250899,0.00595512,0.00406292,
               0.02573248,0.01218337,0.01375329,0.02052385,0.02302154]
gb_feature = [0.03372039,0.04139361,0.01562614,0.03445772,0.01262616,
              0.00948461,0.02323761,0.01194205,0.30467129,0.05419811,
              0.03116018,0.20007032,0.00280088,0.02633729,0.03027627,
              0.00368411,0.00182132,0.00155612,0.0009117,0.01510943,
              0.00659559,0.10958028,0.00368778,0.00252866,0.0225224,]


cols = train.drop("price_t",axis = 1).columns.values
feature_dataframe = pd.DataFrame({'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature})
feature_dataframe["average"] = 0.25 * (
    feature_dataframe["Random Forest feature importances"]+
    feature_dataframe["Extra Trees  feature importances"]+
    feature_dataframe["AdaBoost feature importances"]+
    feature_dataframe["Gradient Boost feature importances"])
feature_dataframe.sort_values("average",ascending = False)




#source: source:https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python#Ensembling-&-Stacking-models
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')



base_predictions = pd.DataFrame({
    'RandomForest': rf_train.ravel(),
    'ExtraTrees': et_train.ravel(),
    'AdaBoost': ada_train.ravel(),
    'GradientBoost': gb_train.ravel(),
    'ground_truth': train_y
    })
base_predictions.head(10)


train_x_second = np.concatenate((et_train, rf_train, ada_train, gb_train), axis=1)
test_x_second = np.concatenate((et_test, rf_test, ada_test, gb_test), axis=1)


X_half_1 = train_x_second[:int(train_x_second.shape[0] / 2)]
X_half_2 = train_x_second[int(train_x_second.shape[0] / 2):]

y_half_1 = train_y[:int(train_y.shape[0] / 2)]
y_half_2 = train_y[int(train_y.shape[0] / 2):]


d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, free_raw_data=False)
d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, free_raw_data=False)

watchlist_1 = [d_half_1, d_half_2]
watchlist_2 = [d_half_2, d_half_1]

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 60,
    "learning_rate": 0.0005,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}

print("Building model with first half and validating on second half:")
model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=10000, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)

print("Building model with second half and validating on first half:")
model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=10000, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)


pred = (np.expm1(model_half_1.predict(test_x_second, num_iteration=model_half_1.best_iteration)) + 88000) / 2
pred += (np.expm1(model_half_2.predict(test_x_second, num_iteration=model_half_2.best_iteration)) + 88000) / 2
ori = np.expm1(test_y) + 88000
res = pd.DataFrame({"ground_truth":ori, "predicted":pred})



# Now we use MAE to evaluate the accuracy of our model.

res["residual"] = np.abs(res["ground_truth"] - res["predicted"])
res["residual"].mean() / res["ground_truth"].mean()
