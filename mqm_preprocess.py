import pandas as pd
from sklearn.model_selection import train_test_split
import csv

def preprocess_wmt_data(train_save_path,test_save_path,test_size=0.2,random_state = 42):
    path_1 = "WMT/wmt-zhen-newstest2020.csv"
    path_2 = "WMT/wmt-zhen-newstest2021.csv"
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    df = pd.concat((df1, df2))
    df = df[['mt', 'ref', 'score']].drop_duplicates()
    trainset, testset = train_test_split(df, test_size=test_size,random_state=random_state)
    sentences_to_corrupt = trainset[['ref']]
    with open(train_save_path, 'w') as f:
        for _,line in sentences_to_corrupt.iterrows():
            f.write(f"{line[0]}\n")
    testset = testset[['mt', 'ref', 'score']]
    csvwriter = csv.writer(open(test_save_path,'w'))
    csvwriter.writerow(['mt', 'ref', 'score'])
    for _,row in testset.iterrows():
        csvwriter.writerow(row)
preprocess_wmt_data('case_study_ref/wmt_train.txt','wmt_test.csv')