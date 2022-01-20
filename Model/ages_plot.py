import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_ages_plot(data):
    pd.crosstab(data.age, data.target).plot(kind="bar", figsize=(20, 6))
    plt.title('Heart Disease Frequency for Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('./plots/heartDiseaseAndAges.png')
    plt.close()
