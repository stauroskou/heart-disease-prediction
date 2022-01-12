from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


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
