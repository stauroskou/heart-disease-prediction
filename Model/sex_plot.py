import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_sex_plot(data):
    sns.countplot(x='sex', data=data, palette="mako_r")
    plt.xlabel("Sex (0 = female, 1= male)")
    plt.savefig("./plots/sex_plot")
    plt.close()
