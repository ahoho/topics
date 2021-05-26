import pandas as pd
from collections import defaultdict
import sys
import json

task_names = ['intrusion', 'ratings']
model_names = ['mallet', 'dvae', 'etm']

def main():

    FOLDER = sys.argv[1]
    SOURCE = sys.argv[2]
    TASK = sys.argv[3]

    if TASK == 'intrusion':
        correct_rating, correct_confidence = '6', '7'
    elif TASK == 'ratings':
        correct_rating, correct_confidence = '3', '4'


    data = json.load(open(f'{FOLDER}/all_data.json', 'r'))#pd.read_json(f'{FOLDER}/all_data.json')
    human_data = pd.read_csv(f'{FOLDER}/{SOURCE}_{TASK}_sampled.csv')
    human_data = human_data.drop(columns = 'Unnamed: 0')

    models_raw = {}
    for model in model_names:
        ordered_cols = {}
        for col in human_data.columns: 
            if (col.split('_')[1]) == model:
                ordered_cols[int(col.rpartition('_')[2])] = human_data[col]
        models_raw[model] = ordered_cols

    #models_final = {'mallet':{}, 'dvae':{}, 'etm':{}}
    models_final = defaultdict(dict)

    #loop through all columns
    for key,val in models_raw.items():
        all_ratings, all_confidences =[],[]
        
        for i in range(0, len(val)):
            ratings, confidences = [], []
            #loop through each annotator in each column
            for j in val[i]:
                rating, confidence = j.split(',')

                #convert qualtrics to binary answer
                if TASK == 'intrusion':
                    if rating == correct_rating:
                        ratings.append(1)
                    else:
                        ratings.append(0)
                elif TASK == 'ratings':
                    ratings.append((int(rating)))

                if confidence == correct_confidence:
                    confidences.append(1)
                else:
                    confidences.append(0)

            all_ratings.append(ratings)
            all_confidences.append(confidences)
        
        models_final[key]['ratings'] = all_ratings
        models_final[key]['confidences'] = all_confidences

    for model in model_names:
        data[f'{SOURCE}'][model]['metrics'][f'{TASK}_scores_raw'] = models_final[model]['ratings']
        data[f'{SOURCE}'][model]['metrics'][f'{TASK}_scores_avg'] = [sum(row)/len(row) for row in models_final[model]['ratings']]
        data[f'{SOURCE}'][model]['metrics'][f'{TASK}_confidences_raw'] = models_final[model]['confidences']
        data[f'{SOURCE}'][model]['metrics'][f'{TASK}_confidences_avg'] = [sum(row)/len(row) for row in models_final[model]['confidences']]


    json.dump(data, open(f'{FOLDER}/all_data.json', 'w'))#data.to_json(f'{FOLDER}/all_data.json')


if __name__ == "__main__":
    main()
