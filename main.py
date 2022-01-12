from flask.wrappers import Request
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import *
from flask import request as flask_request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

data = pd.read_csv("./Data/heart.csv")
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


def print_score(clf, X_test, y_test):
    pred = clf.predict(X_test)
    clf_report = pd.DataFrame(
        classification_report(y_test, pred, output_dict=True))
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

rf_clf = KMeans(n_clusters=2, random_state=2)
rf_clf.fit(X_train, y_train)

filename = './saved_models/finalized_model.sav'
pickle.dump(rf_clf, open(filename, 'wb'))


# [0] = 'age', [1] = 'trestbps', [2] = 'chol', [3] = 'thalach', [4] = 'oldpeak', [5] = 'sex_0', [6] = 'sex_1', [7] = 'cp_0', [8] = 'cp_1', [9] = 'cp_2', [10] = 'cp_3', [11] = 'fbs_0',
# [12] = 'fbs_1', [13] = 'restecg_0', [14] = 'restecg_1', [15] = 'restecg_2', [16] = 'exang_0', [17] = 'exang_1', [18] = 'slope_0', [19] = 'slope_1', [20] = 'slope_2', [21] = 'ca_0', [22] = 'ca_1',
#  [23] = 'ca_2', [24] = 'ca_3', [25] = 'ca_4', [26] = 'thal_0', [27] = 'thal_1', [28] = 'thal_2', [29] = 'thal_3'
# testarray2 = np.zeros((1, 30))
# testDict = {}

print_score(rf_clf, X_test, y_test)
# print(prediction[0])

@app.route("/getresults", methods=['GET'])
def getresults():
    age: float = float(flask_request.args.get('age'))
    sex: float = float(flask_request.args.get('sex'))
    cp: float = float(flask_request.args.get('cp'))
    trestbps: float = float(flask_request.args.get('trestbps'))
    chol: float = float(flask_request.args.get('chol'))
    fbs: float = float(flask_request.args.get('fbs'))
    restecg: float = float(flask_request.args.get('restecg'))
    thalach: float = float(flask_request.args.get('thalach'))
    exang: float = float(flask_request.args.get('exang'))
    oldpeak: float = float(flask_request.args.get('oldpeak'))
    slope: float = float(flask_request.args.get('slope'))
    ca: float = float(flask_request.args.get('ca'))
    thal: float = float(flask_request.args.get('thal'))

    testarray2 = np.zeros((1, 30))
    testDict = {}
    testDict.update({"sex": sex,
                 "cp": cp, "fbs": fbs, "restecg": restecg,
                 "exang": exang, "slope": slope, "ca": ca, "thal": thal})
    testarray2[0, 0] = age
    testarray2[0, 1] = trestbps
    testarray2[0, 2] = chol
    testarray2[0, 3] = thalach
    testarray2[0, 4] = oldpeak
    for x in testDict.keys():
        if(x == "sex"):
            if(testDict[x] == 0):
                testarray2[0, 5] = 1
            else:
                testarray2[0, 6] = 1
        elif(x == "cp"):
            if(testDict[x] == 0):
                testarray2[0, 7] = 1
            elif(testDict[x] == 1):
                testarray2[0, 8] = 1
            elif(testDict[x] == 2):
                testarray2[0, 9] = 1
            else:
                testarray2[0, 10] = 1
        elif(x == "fbs"):
            if(testDict[x] == 0):
                testarray2[0, 11] = 1
            else:
                testarray2[0, 12] = 1
        elif(x == "restecg"):
            if(testDict[x] == 0):
                testarray2[0, 13] = 1
            elif(testDict[x] == 1):
                testarray2[0, 14] = 1
            else:
                testarray2[0, 15] = 1
        elif(x == "exang"):
            if(testDict[x] == 0):
                testarray2[0, 16] = 1
            else:
                testarray2[0, 17] = 1
        elif(x == "slope"):
            if(testDict[x] == 0):
                testarray2[0, 18] = 1
            elif(testDict[x] == 1):
                testarray2[0, 19] = 1
            else:
                testarray2[0, 20] = 1
        elif(x == "ca"):
            if(testDict[x] == 0):
                testarray2[0, 21] = 1
            elif(testDict[x] == 1):
                testarray2[0, 22] = 1
            elif(testDict[x] == 2):
                testarray2[0, 23] = 1
            elif(testDict[x] == 3):
                testarray2[0, 24] = 1
            elif(testDict[x] == 4):
                testarray2[0, 25] = 1
        elif(x == "thal"):
            if(testDict[x] == 0):
                testarray2[0, 26] = 1
            elif(testDict[x] == 1):
                testarray2[0, 27] = 1
            elif(testDict[x] == 2):
                testarray2[0, 28] = 1
            elif(testDict[x] == 3):
                testarray2[0, 29] = 1

    columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',  'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
               'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2', 'ca_0',
               'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
    dfTest = pd.DataFrame(data=testarray2, columns=columns)
    dfTest[col_to_scale] = qq.transform(dfTest[col_to_scale])
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction = loaded_model.predict(dfTest)
    return jsonify({"Prediction": str(prediction[0])})

if __name__ == "__main__":
    app.run()