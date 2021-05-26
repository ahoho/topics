import pandas as pd
import regex as re
import sys


def rename_columns(df):
    #drop extra row
    df = df.drop(axis=0, index=[0,1])
    new_index = {}
    regex = re.compile(f'(.*)_\d+')
    wiki_typo = re.compile(f'wiki_mallet*')
    
    for i in df.columns:
        if regex.match(i):
            if wiki_typo.match(i):
                new_name = i.replace("wiki_", "wikitext_")
                new_name = new_name.rpartition('_')[0]
            else:
                new_name = i.rpartition('_')[0]
            new_index[i] = new_name
    df = df.rename(columns=new_index)
    return df

def main():

    #specify location + wikitext/nytimes
    FOLDER = sys.argv[1]
    SOURCE = sys.argv[2]

    #4 splits have to be read in
    data1 = pd.read_csv(f'{FOLDER}/{SOURCE}_intrusion_1.csv')
    data2 = pd.read_csv(f'{FOLDER}/{SOURCE}_intrusion_2.csv')
    data3 = pd.read_csv(f'{FOLDER}/{SOURCE}_intrusion_3.csv')
    data4 = pd.read_csv(f'{FOLDER}/{SOURCE}_intrusion_4.csv')

    #get all columns that have underscore followed by number and rename them.  drop extra row 
    data1 = rename_columns(data1)
    data2 = rename_columns(data2)
    data3 = rename_columns(data3)
    data4 = rename_columns(data4)

    df_inner = pd.concat([data1, data2, data3, data4], axis=0,  ignore_index=True)

    df_inner = df_inner.drop(columns = ['Duration (in seconds)', 'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Finished', 
                             'RecordedDate', 'RecipientLastName', 'RecipientFirstName', 'ResponseId',
                             'RecipientEmail', 'ExternalReference', 'LocationLatitude', 'LocationLongitude',
                             'DistributionChannel', 'UserLanguage', 'prolific', 'PROLIFIC_PID'
                            ])

    #inconsitent name
    if SOURCE == "nyt":
        SOURCE = "nytimes"
        
    df_inner.to_csv(f'{FOLDER}/{SOURCE}_intrusion.csv')


if __name__ == "__main__":
    main()


