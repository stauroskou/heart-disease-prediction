import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_target_plot(data):
    sns.countplot(x="target", data=data, palette="bwr")
    plt.savefig("./plots/target_plot")
    plt.close()
