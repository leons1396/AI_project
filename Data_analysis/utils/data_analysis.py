import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def encoding_labels(df, col_name, encoding):
    """
    Encodes the labels
    :param df: dataset
    :type df: pd.DataFrame
    :param col_name: column name to encode
    :type: str
    :param encoding: encoding values in ascending order
    :type: list
    :return: encoded dataset
    :type: pd.DataFrame
    """
    df.loc[df[col_name] == 'Karotte', col_name] = encoding[0]  #0
    df.loc[df[col_name] == 'Kartoffel', col_name] = encoding[1]  #1
    df.loc[df[col_name] == 'Zwiebel', col_name] = encoding[2]  #2
    df.loc[df[col_name] == 'Karotte_Trieb', col_name] = encoding[3]  #3
    df.loc[df[col_name] == 'Kartoffel_Trieb', col_name] = encoding[4]  #4
    df.loc[df[col_name] == 'Zwiebel_Trieb', col_name] = encoding[5]  #5
    return df

def get_df_with_two_labels(df, label_1, label_2):
    return df.loc[(df["Classes"] == label_1) | (df["Classes"] == label_2)]