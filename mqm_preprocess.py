import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from unidecode import unidecode
from tqdm import tqdm


def fix_to_ascii(df):
    problem_rows = set()
    not_problems_rows = set()
    new_df = []
    for row, x in tqdm(df.iterrows()):
        row_dict = {'score': x['score'], 'mt': None, 'ref': None}
        flag = True
        for option in ['mt', 'ref']:
            sen = x[option]
            new_sen = []
            for char in sen:
                if ord(char) > 128:
                    new_char = unidecode(char)
                    if len(new_char) != 1 or ord(new_char) > 128:
                        problem_rows.add(row)
                        flag = False
                        break
                    else:
                        new_sen.append(new_char)
                else:
                    new_sen.append(char)
            new_sen = "".join(new_sen)
            row_dict[option] = new_sen
        if flag:
            not_problems_rows.add(row)
            new_df.append(tuple(row_dict.values()))
    df = pd.DataFrame.from_records(new_df, columns=['score', 'mt', 'ref'])
    return df,problem_rows,not_problems_rows
def preprocess_wmt_data(train_save_path,test_save_path,test_size=0.2,random_state = 42):
    path_1 = "WMT/wmt-zhen-newstest2020.csv"
    path_2 = "WMT/wmt-zhen-newstest2021.csv"
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    df = pd.concat((df1, df2))
    df = df[['mt', 'ref', 'score']].drop_duplicates().dropna().reset_index(drop=True)
    df,_,_ = fix_to_ascii(df)
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
def small_test(train_save_path,test_save_path,test_size=0.2,random_state = 42,train_lines = 6000):
    path_1 = "WMT/wmt-zhen-newstest2020.csv"
    path_2 = "WMT/wmt-zhen-newstest2021.csv"
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)
    df = pd.concat((df1, df2))
    df = df[['mt', 'ref', 'score']].drop_duplicates().dropna().reset_index(drop=True)
    df,_,_ = fix_to_ascii(df)
    trainset, testset = train_test_split(df, test_size=test_size,random_state=random_state)
    sentences_to_corrupt = trainset[['ref']][:train_lines]
    with open(train_save_path, 'w') as f:
        for index,line in sentences_to_corrupt.iterrows():
            f.write(f"{line[0]}\n")
    small_test = pd.concat((trainset.iloc[train_lines:],testset))
    small_test = small_test[['mt', 'ref', 'score']]
    csvwriter = csv.writer(open(test_save_path,'w'))
    csvwriter.writerow(['mt', 'ref', 'score'])
    for _,row in small_test.iterrows():
        csvwriter.writerow(row)

# preprocess_wmt_data('case_study_ref/wmt_train_fixed.txt','test/wmt_test_fixed.csv')
# preprocess_wmt_data('case_study_ref/wmt_train.txt','wmt_test.csv')
# small_test('case_study_ref/wmt_train_small_fixed.txt','test/wmt_test_small_fixed.csv')
