"""
This script will generate a topics 'ratings' survey based on a topics survey
"""
import argparse
import datetime
import json
import random

from survey_utils import (
    load_topics_file, format_survey_blocks,
    update_survey_blocks_element, load_multi_topics_file,
    set_redirect_url, set_nytimes_dataset, set_wikitext_dataset,
    load_fake_topics)


def format_ratings_question(question_id, topic_id, terms, model_name="topics", top_terms=10, include_confidence=False):
    """
    NOTE: The last value in terms should be the intruder word
    """
    data_export_tag = f"{model_name}_{topic_id}"
    terms_strings = ", ".join(terms[:top_terms])
    
    ratings_question = {
        'SurveyID': 'SV_4My2xeJALXVC80l',
        'Element': 'SQ',
        'PrimaryAttribute':  f'QID{question_id}',
        'SecondaryAttribute': f'How related are the following terms: {terms_strings}',
        'TertiaryAttribute': None,
        'Payload': {'QuestionText': f'<span style="font-weight: bolder;">How related are the following terms:</span><div><b>{terms_strings}</b></div>',
            'DataExportTag': data_export_tag,
            'QuestionType': 'MC',
            'Selector': 'SAHR',
            'SubSelector': 'TX',
            'DataVisibility': {'Private': False, 'Hidden': False},
            'Configuration': {'QuestionDescriptionOption': 'UseText',
            'LabelPosition': 'BELOW'},
            'QuestionDescription': f'How related are the following terms: {terms_strings}',
            'Choices': {'1': {'Display': 'Not Very Related'},
            '2': {'Display': 'Somewhat Related'},
            '3': {'Display': 'Very Related'}},
            'ChoiceOrder': ['3', '2', '1'],
            'Validation': {'Settings': {'ForceResponse': 'ON',
            'ForceResponseType': 'ON',
            'Type': 'None'}},
            'Language': [],
            'NextChoiceId': 4,
            'NextAnswerId': 1,
            'QuestionID': f'QID{question_id}'
            }
        }

    ratings_question_confidence = {
        "SecondaryAttribute": "Please rate how related the following terms are to each other and how familiar    you are in your a...", 
        "TertiaryAttribute": None, 
        "Element": "SQ", 
        "SurveyID": "SV_cGgwR9yoWhFLjSK", 
        "Payload": {
            "QuestionType": "MC", 
            "QuestionID": f'QID{question_id}', 
            "Validation": {
            "Settings": {
                "ForceResponseType": "ON", 
                "MaxChoices": "2", 
                "Type": "MinChoices", 
                "ForceResponse": "ON", 
                "MinChoices": "2"
            }
            }, 
            "QuestionText": f"<div>Please rate how related the following terms are to each other and how familiar you are with the terms:</div><div><br><div> {terms_strings} </div></div>", 
            "Language": [], 
            "NextChoiceId": 7, 
            "GradingData": [], 
            "NextAnswerId": 1, 
            "Selector": "MAVR", 
            "QuestionDescription": "Please rate how related the following terms are to each other and how familiar you are in your a...", 
            "DataVisibility": {
            "Hidden": False, 
            "Private": False
            }, 
            "DefaultChoices": False, 
            "ChoiceOrder": [
            1, 
            2, 
            3, 
            4, 
            5,
            6
            ], 
            "SubSelector": "TX", 
            "DataExportTag": data_export_tag, 
            "Choices": {
            "1": {
                "Display": "Not very related"
            }, 
            "3": {
                "Display": "Very related"
            }, 
            "2": {
                "Display": "Somewhat related"
            }, 
            "4": {
                "Display": "I am familiar with most of these terms."
            },
            "5": {
                "Display": "I am <em><strong>not</strong></em> familiar with most of these terms, but I <em><strong>can</strong></em> answer confidently."
            },
            "6": {
                "Display": "I am <em><strong>not</strong></em> familiar with most of these terms, and so I <em><strong>cannot</strong></em> answer confidently."
            } 
            }, 
            "Configuration": {
            "QuestionDescriptionOption": "UseText"
            }, 
            "ChoiceGroups": {
            "cg_2": {
                "GroupLabel": "Answer Confidence", 
                "Options": {
                "Selection": "SAWithinGroup"
                }, 
                "ChoiceGroupOrder": [
                4, 
                5,
                6
                ]
            }, 
            "cg_1": {
                "Randomization": {
                "Type": "None"
                }, 
                "GroupLabel": "Rating", 
                "Options": {
                "Selection": "SAWithinGroup"
                }, 
                "ChoiceGroupOrder": [
                3, 
                2, 
                1
                ]
            }
            }, 
            "ChoiceGroupOrder": [
            "cg_1", 
            "cg_2"
            ]
        }, 
        "PrimaryAttribute": f'QID{question_id}'
    }
    
    return ratings_question_confidence if include_confidence else ratings_question

def gap_filter_questions(dataset, model_type, topics):
    # filters a set of topics to keep only the ones needed for the 'gap' ratings
    gap_topics = {
        "nytimes": {
            "mallet": [34, 48, 5, 20],
            "dvae": [0, 28, 27, 12, 38],
            "etm": [5, 22, 1, 3, 34, 12, 27]
        },
        "wikitext": {
            "mallet": [3, 19, 24, 23, 5],
            "dvae": [33],
            "etm": [24]
        }
    }
    
    new_set = []
    for topic in topics:
        if "mallet" in model_type:
            if topic["topic_id"] in gap_topics[dataset]["mallet"]:
                new_set.append(topic)
        elif "dvae" in model_type:
            if topic["topic_id"] in gap_topics[dataset]["dvae"]:
                new_set.append(topic)
        elif "etm" in model_type: 
            if topic["topic_id"] in gap_topics[dataset]["etm"]:
                new_set.append(topic)

    return new_set


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--file", dest="file", help="File path for input files", required=True)
    arg_parser.add_argument("-o", "--output", dest="output", help="Output file name", required=True)
    arg_parser.add_argument("-s", "--survey_name", dest="survey_name", help="Survey name", required=True)
    arg_parser.add_argument("-n", "--num_topics", dest="num_topics",
                            type=int, help="Number of Topics to sample",
                            default=20)
    arg_parser.add_argument("-m", "--model_name", dest="model_name", help="Useful model name param", default="topics")
    arg_parser.add_argument("--include_confidence", dest="include_confidence", action="store_true", default=False, help="Include an option specify confidence")
    arg_parser.add_argument("--multi", dest="multi", help="Multi topic file experiment", default=None)
    arg_parser.add_argument("--file_2", dest="file_2", help="Second topic file")
    arg_parser.add_argument("--model_name_2", dest="model_name_2", help="Useful model name param for the second topic file")
    arg_parser.add_argument("--num_rand_questions", default="25", help="Number of questions to show out of the max")
    arg_parser.add_argument("--prolific", help="URL of the link back to Prolific")
    arg_parser.add_argument("--file_3", dest="file_3", help="Third topic file")
    arg_parser.add_argument("--model_name_3", dest="model_name_3", help="Useful model name param for the third topic file")
    arg_parser.add_argument("--gap", dest="gap", help="Create Gap analysis survey only", default=False, action="store_true")

    args = arg_parser.parse_args()

    if args.prolific:
        survey_template = json.load(open("Topic_Ratings_Template_Prolific.qsf"))
    else:    
        survey_template = json.load(open("Topic_Ratings_Template.qsf"))

    # We need to set an appropriate offset for the questions depending on how many parts of the introduction
    # we have
    question_id = 3
    if args.prolific:
        question_id = 4


    questions = []

    if args.multi:
        topics_1 = load_multi_topics_file(args.file, args.model_name)[args.multi]
        topics_2 = load_multi_topics_file(args.file_2, args.model_name_2)[args.multi]
        topics_3 = load_multi_topics_file(args.file_3, args.model_name_3)[args.multi]
        fake_topics = load_fake_topics()

        selected_topics_idxs_1 = random.sample(range(len(topics_1)), args.num_topics)
        selected_topics_idxs_2 = random.sample(range(len(topics_2)), args.num_topics)
        selected_topics_idxs_3 = random.sample(range(len(topics_3)), args.num_topics)

        if args.gap:
            topics_1 = gap_filter_questions(args.multi, args.model_name, topics_1)
            topics_2 = gap_filter_questions(args.multi, args.model_name_2, topics_2)
            topics_3 = gap_filter_questions(args.multi, args.model_name_3, topics_3)
            fake_topics = fake_topics[:2]

            selected_topics_idxs_1 = random.sample(range(len(topics_1)), len(topics_1))
            selected_topics_idxs_2 = random.sample(range(len(topics_2)), len(topics_2))
            selected_topics_idxs_3 = random.sample(range(len(topics_3)), len(topics_3))

        
        print(f"Selected the following topics for ratings from model 1: {selected_topics_idxs_1}")
        print(f"Selected the following topics for ratings from model 2: {selected_topics_idxs_2}")
        print(f"Selected the following topics for ratings from model 2: {selected_topics_idxs_3}")

        for topic_idx in selected_topics_idxs_1:
            topic = topics_1[topic_idx]
            questions.append(
                format_ratings_question(
                    question_id=question_id,
                    topic_id=topic["topic_id"],
                    terms=topic["terms"],
                    model_name=args.model_name,
                    include_confidence=args.include_confidence
                )
            )
            question_id += 1

        for topic_idx in selected_topics_idxs_2:
            topic = topics_2[topic_idx]
            questions.append(
                format_ratings_question(
                    question_id=question_id,
                    topic_id=topic["topic_id"],
                    terms=topic["terms"],
                    model_name=args.model_name_2,
                    include_confidence=args.include_confidence
                )
            )
            question_id += 1

        for topic_idx in selected_topics_idxs_3:
            topic = topics_3[topic_idx]
            questions.append(
                format_ratings_question(
                    question_id=question_id,
                    topic_id=topic["topic_id"],
                    terms=topic["terms"],
                    model_name=args.model_name_3,
                    include_confidence=args.include_confidence
                )
            )
            question_id += 1

        for topic in fake_topics:
            questions.append(
                format_ratings_question(
                    question_id=question_id,
                    topic_id=topic["topic_id"],
                    terms=topic["terms"],
                    model_name=topic["model_name"],
                    include_confidence=args.include_confidence
                )
            )
            question_id += 1

    else:
        all_topics = load_topics_file(args.file)
        
        selected_topics_idxs = random.sample(range(len(all_topics)), args.num_topics)
        print(f"Selected the following topics for ratings: {selected_topics_idxs}")

        for topic_idx in selected_topics_idxs:
            topic = all_topics[topic_idx]
            questions.append(
                format_ratings_question(
                    question_id=question_id,
                    topic_id=topic["topic_id"],
                    terms=topic["terms"],
                    model_name=args.model_name,
                    include_confidence=args.include_confidence
                )
            )
            question_id += 1

    print(f"Generating {len(questions)} questions")

    if survey_template["SurveyElements"][0]["PrimaryAttribute"] == "Survey Blocks":
        update_survey_blocks_element(
            survey_template["SurveyElements"][0], 
            questions,
            num_random=args.num_rand_questions
        )
        #survey_template["SurveyElements"][0]["PrimaryAttribute"]
        
    # Step 2: Wipe the existing old questions that are there for easy reference
    idx_to_delete = []
    for idx, element in enumerate(survey_template["SurveyElements"]):

        if element["Element"] == "SQ":
            if element.get("Payload", {}).get('QuestionType') == "MC":
                idx_to_delete.append(idx)

    survey_template["SurveyElements"] = [element for idx, element in enumerate(survey_template["SurveyElements"])
                                        if not idx in idx_to_delete]
        
    # Step 3: Append the new questions to the end
    for question in questions:
        survey_template["SurveyElements"].append(question)

    day = datetime.date.today().strftime('%Y-%m-%d')
    survey_template["SurveyEntry"]["SurveyName"] = f"Ratings {args.survey_name.title()} {day}"

    if args.prolific:
        set_redirect_url(survey_template["SurveyElements"], args.prolific)

    if args.multi == "nytimes":
        set_nytimes_dataset(survey_template["SurveyElements"])
    if args.multi == "wikitext":
        set_wikitext_dataset(survey_template["SurveyElements"])

    json.dump(survey_template,
            open(f"{args.output}.qsf", "w+"),
            indent=2)
    


    