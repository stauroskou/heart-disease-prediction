from Model.ages_plot import create_ages_plot
from Model.printScore import print_score
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from Model.sex_plot import create_sex_plot

from Model.target_plot import create_target_plot


def createModel():
    data = pd.read_csv("./Data/heart.csv")
    create_sex_plot(data)
    create_target_plot(data)
    create_ages_plot(data)

    categorical_val = []
    continous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continous_val.append(column)

    categorical_val.remove("target")
    dataset = pd.get_dummies(data, columns=categorical_val)

    qq = QuantileTransformer()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[col_to_scale] = qq.fit_transform(dataset[col_to_scale])
    X = dataset.drop('target', axis=1)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)

    rf_clf = KMeans(n_clusters=2, random_state=2)
    rf_clf.fit(X_train, y_train)
    filename = './saved_models/finalized_model.sav'
    pickle.dump(rf_clf, open(filename, 'wb'))
    print_score(rf_clf, X_test, y_test)
