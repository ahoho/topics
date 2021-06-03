
import csv
import json

"""
{<dataset>: {
    <model>: {
        intrusion: {
               <topic_0_scores> : [0, 1, 1, ... # up to num annotators
               <topic_1_scores> : [1, 0, ...
               ...
               <topic_0_word_knowledge>: [0, 1, ...
         }
         ratings: {
               <topic_0_scores> : [1, 2, ...
               ...
               <topic_0_word_knowledge>: [1, 1, ...
"""

# set of users (by prolific id) we want to eliminate
PROLIFIC_PID_BLACKLIST = {

}

if __name__ == "__main__":

    topics_files = {
        "../may_22_2021/mallet-topics.json",
        "../may_22_2021/dvae-topics.json",
        "../may_22_2021/etm-topics.json"
    }
    all_topics = []
    for t_file in topics_files:
        topics_file = json.load(open(t_file))
        for dataset in topics_file:
            
            for idx, topic in enumerate(topics_file[dataset]["topics"]):
                all_topics.append({
                    "dataset": dataset,
                    "topic_id": idx,
                    "terms": topic,
                    "model": t_file.replace("-topics.json", "").replace("../may_22_2021/", "")
                })

    
    print(len(all_topics))
    all_topics_index = {
        f"{topic['dataset']}_{topic['model']}_{topic['topic_id']}": topic for topic in all_topics
    }

    coherences = json.load(open("../may_22_2021/coherences.json"))
    #print(coherences)
    for dataset in coherences:
        for model in coherences[dataset]:
            for metric in coherences[dataset][model]["metrics"]:
                for idx, value in enumerate(coherences[dataset][model]["metrics"][metric]):
                    all_topics_index[f"{dataset}_{model}_{idx}"][metric] = value
                    
                    assert all_topics_index[f"{dataset}_{model}_{idx}"]["terms"][:20] == coherences[dataset][model]["topics"][idx]
    print(all_topics_index)
    json.dump(all_topics_index, open("../may_22_2021/combined_intrusion_coherence_terms.json", "w+"))
    quit()

    intrusion_setup_files = [
        "../may_22_2021/wiki_intrusion_1_wiki_mallet.json",
        "../may_22_2021/wiki_intrusion_2_wiki_mallet.json",
        "../may_22_2021/wiki_intrusion_3_wiki_mallet.json",
        "../may_22_2021/wiki_intrusion_4_wiki_mallet.json"
    ]

    


    intrusion_files = [
        "../may_22_2021/wikitext_intrusion_1.csv",
        "../may_22_2021/wikitext_intrusion_2.csv"
    ]

    data_variants = [
        [row for row in csv.DictReader(open(_file))] for _file in intrusion_files
    ]
    # group things by model type
    models = {
        "mallet": {
            "intrusion": {},
            "ratings": {}
        },
        "dvae": {
            "intrusion": {},
            "ratings": {}
        },
        "etm": {
            "intrusion": {},
            "ratings": {}
        }
    }

    dataset_name = "wikitext"
    for variant in data_variants:
        
        row_count = 0
        for row in variant:
            if row_count < 2:
                row_count += 1
                continue
            row_count += 1
            
            if row["PROLIFIC_PID"] in PROLIFIC_PID_BLACKLIST:
                print(f"Skipping user: {row['PROLIFIC_PID'] }")
                continue
            for column in row:
                if dataset_name in column:
                    
                    # skip any columns without an answer
                    if not row[column]:
                        continue
                    question_name_components = column.split("_")
                    question_dataset = question_name_components[0]
                    question_model = question_name_components[1]
                    question_topic_id = question_name_components[2]
                    if question_topic_id not in models[question_model]["intrusion"]:
                        models[question_model]["intrusion"][question_topic_id] = {
                            "topic_id": question_topic_id,
                            "responses": [],
                            "confidence": []
                        }
                    answer_parts = row[column].split(",")
                    
                    models[question_model]["intrusion"][question_topic_id]["responses"].append(answer_parts[0])
                    models[question_model]["intrusion"][question_topic_id]["confidence"].append(answer_parts[1])
                    
                    print(models[question_model]["intrusion"][question_topic_id])