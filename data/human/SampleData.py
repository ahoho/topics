import pandas as pd
import regex as re
import sys

def main():
    FOLDER = sys.argv[1]
    SOURCE = sys.argv[2]
    TASK =  sys.argv[3]

    data = pd.read_csv(f'{FOLDER}/{SOURCE}_{TASK}.csv')
    data = data.drop(columns = ['Unnamed: 0'])

    new_df = pd.DataFrame()
    col_list = []

    for i, col in enumerate(data.columns):
        #TEMPORARILY WHILE RATINGS IS LESS THAN 15
        if TASK == 'ratings':          
            new_df[col] = data[col].dropna().sample(15,random_state=1, replace = True).values
        elif TASK == 'intrusion':
            new_df[col] = data[col].dropna().sample(15,random_state=1).values


    new_df.to_csv(f'{FOLDER}/{SOURCE}_{TASK}_sampled.csv')

if __name__ == "__main__":
    main()
