import pandas as pd
import regex as re
import sys

def main():

    #specify location + wikitext/nytimes
    FOLDER = sys.argv[1]
    SOURCE = sys.argv[2]

    #4 splits have to be read in
    data = pd.read_csv(f'{FOLDER}/{SOURCE}_ratings.csv')
    data = data.drop(axis=0, index=[0,1])
    data = data.drop(columns = ['Duration (in seconds)', 'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Finished', 
                             'RecordedDate', 'RecipientLastName', 'RecipientFirstName', 'ResponseId',
                             'RecipientEmail', 'ExternalReference', 'LocationLatitude', 'LocationLongitude',
                             'DistributionChannel', 'UserLanguage', 'prolific', 'PROLIFIC_PID'
                            ])
    #inconsitent name
    if SOURCE == "nyt":
        SOURCE = "nytimes"
        
    
    data.to_csv(f'{FOLDER}/{SOURCE}_ratings.csv')


if __name__ == "__main__":
    main()


