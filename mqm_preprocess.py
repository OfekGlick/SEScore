import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_wmt_data(test_size = 0.2):
    path_1 = "Data/wmt-zhen-newstest2020.csv"
    path_2 = "Data/wmt-zhen-newstest2021.csv"
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    df = pd.concat((df1,df2))
    df = df[['mt','ref','score']].drop_duplicates()
    trainset,testset = train_test_split(df,test_size=test_size)
    sentences_to_corrupt = trainset[['ref']]
    testset = testset[['mt','ref','score']]
    return sentences_to_corrupt,testset



