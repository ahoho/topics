"""
Script for generating a Word Intrusion Qualtrics Experiment

Input should be a file of topics in the following form:
- 1 Topic per line
- Topic terms are in descending order of model weights (highest weights at the beginning of the line)
- Terms are comman separated
- NOTE: We will use the 'line index' value for the topic ids in order to preserve some record of which topics
        are being analyzed
"""

import argparse
import datetime
import json
import random

from survey_utils import (
    load_topics_file, format_survey_blocks, load_multi_topics_file,
    set_redirect_url, set_nytimes_dataset, set_wikitext_dataset,
    update_survey_blocks_element, load_fake_topics
)

def load_topics_file(filepath, delimiter=","):
    reader = open(filepath)
    lines = [line.replace("\t", " ") for line in reader.readlines()]
    if len(lines[0].split(" ")) > len(lines[0].split(",")):
        delimiter = " "
    #print(lines)
    # Otherwise we use the regular comma delimiter
    #print("Delimiter: ", delimiter, "---")
    topics = []
    for idx, line in enumerate(lines):
        topics.append(
            {
                "topic_id": idx,
                "terms": line.replace("\n", "").split(delimiter)
            }
        )
    return topics

def setup_word_intrusion(topics_list, n=20, topic_idxs=None, num_terms=5, sample_top_topic_terms=False):
    """
    topics_list: format is a list of dicts [
        {"topic_id": 1, "terms": ["a","b","c"]},
        {"topic_id": 2, "terms": ["a","b","c"]}
    ]
    This is the format given out of the gensim wrapper for Mallet
    
    n: The number of topics to sample
    """
    # Can't sample more than the available topics
    assert len(topics_list) >= n
  
    intruder_list = []
    # Generate n random ints for the selection of topics we'll conduct intrusion on
    if not topic_idxs:
        topic_idxs = random.sample(range(len(topics_list)), n)
        
    selected_intruders = set()
    for topic_idx in topic_idxs:
        
        # select another topic from which to grab a term, exclude the current topic
        random_topic_idx = random.choice([idx for idx in range(0, len(topics_list)) if (idx != topic_idx and idx not in selected_intruders)])
        selected_intruders.add(random_topic_idx)
        # take the top 5 words of the current topic and ONE of the top 5 terms from the top of the other topic
        # assert that the new word is not in the top 50 words of the original topic
        correct_words = [word for word in topics_list[topic_idx]["terms"][:num_terms]]
        
        # This collects the top 50 words of the current topic
        top_topic_words = [word for word in topics_list[topic_idx]["terms"][:50]]

        # This collects the top words of the 'intruder' topics that do NOT overlap with any of the top
        # 10 words of the other topic
        if sample_top_topic_terms:
            top_random_words = random.sample([word for word in topics_list[random_topic_idx]["terms"][:10] \
                                if word not in top_topic_words], num_terms)
        else:
            top_random_words = [word for word in topics_list[random_topic_idx]["terms"][:5] \
                                if word not in top_topic_words]
        
        # EDGE-CASE - The top 50 words of the selected topic may overlap heavily with the
        # 'intruder' topics's top words. In this case, narrow down the set of excluded terms
        # for the current topic to just the top 10. If that doesn't work, then..... skip??
        if not top_random_words:
            top_topic_words = [word for word in topics_list[topic_idx]["terms"][:10]]
            top_random_words = [word for word in topics_list[random_topic_idx]["terms"][:5] \
                                if word not in top_topic_words]
        
            if not top_random_words:
                print(f"Skipping word intrusion for topic {topic_idx} with intruder {random_topic_idx}")
                continue
        # select the intruder word
        selected_intruder = random.choice(top_random_words)
        
        #print(topic_idx, random_topic_idx, correct_words + [selected_intruder])
        
        # The last word in each list is the 'intruder', this should be randomized before showing
        #[topics_list[topic_idx]["topic_id"]] + correct_words + [selected_intruder]
        intruder_list.append(
            {
                "topic_id": topics_list[topic_idx]["topic_id"],
                "intruder_id": topics_list[random_topic_idx]["topic_id"],
                "intruder_term": selected_intruder,
                "topic_terms": correct_words
            }
            )
    return intruder_list

def format_intruder_question(question_id, topic_id, topic_intruder_id, terms, model_name="topics", include_confidence=False):
    """
    NOTE: The last value in terms should be the intruder word
    """
    data_export_tag = f"{model_name}_{topic_id}_{topic_intruder_id}"

        #{'0': {'Display': 'Andrew'}, '1': {'Display': 'Eric'}, '2': {'Display': 'Claire'}, '3': {'Display': 'Riley'},
        #'4': {'Display': 'Daniel'}, '5': {'Display': 'Potato'}}
    choices = {str(idx + 1): {"Display": term} for idx,term in enumerate(terms)}

    if include_confidence:
        len_choice = len(choices)
        choices[str(len_choice + 1)] = {
          "Display": "I am familiar with most of these terms."
        }
        choices[str(len_choice + 2)] = {
          "Display": "I am <em><strong>not</strong></em> familiar with most of these terms, but I <em><strong>can</strong></em> answer confidently."
        }
        choices[str(len_choice + 3)] = {
          "Display": "I am <em><strong>not</strong></em> familiar with most of these terms, and so I <em><strong>cannot</strong></em> answer confidently."
        }
        
    choice_order = [i + 1 for i in range(0, len(choices))]
    choice_order_intrusion = choice_order[:-3]
    choice_order_confidence = choice_order[-3:]

    intruder_question = {
        "SurveyID": "SV_5sXmuibskKlpHmJ",
        "Element": "SQ",
        "PrimaryAttribute": f'QID{question_id}',
        "SecondaryAttribute": "This survey will ask you to evaluate the outputs of a Machine Learning computer model.\u00a0Researcher..",
        "TertiaryAttribute": None,
        'Payload': {
            'QuestionText': 'Identify which word does not belong with the others',
            'DefaultChoices': False,
            'DataExportTag': data_export_tag,
            'QuestionType': 'MC',
            'Selector': 'SAVR',
            'SubSelector': 'TX',
            'DataVisibility': {'Private': False, 'Hidden': False},
            'Configuration': {'QuestionDescriptionOption': 'UseText'},
            'QuestionDescription': 'Identify which word does not belong with the others',
            'Choices': choices,
            'ChoiceOrder': choice_order,
            'Randomization': {'Advanced': None, 'Type': 'All', 'TotalRandSubset': ''},
            'Validation': {'Settings': {'ForceResponse': 'ON',
                'ForceResponseType': 'ON',
                'Type': 'None'}},
            'GradingData': [],
            'Language': [],
            'NextChoiceId': len(choices) + 1,
            'NextAnswerId': 1,
            'QuestionID': f'QID{question_id}'}
    }

    intruder_question_confidence = {
        "SecondaryAttribute": "Please select which term is the least related to all other terms and your...", 
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
          "QuestionText": "Please select which term is the least related to all other terms and your familiarity with the words", 
          "Language": [], 
          "NextChoiceId": len(choices) + 1, 
          "DataVisibility": {
            "Hidden": False, 
            "Private": False
          }, 
          "NextAnswerId": 1, 
          "Selector": "MAVR", 
          "QuestionDescription": "Please select which term is the least related to all other terms and your...", 
          "Randomization": {
            "TotalRandSubset": "", 
            "Type": "None", 
            "Advanced": None
          }, 
          "ChoiceOrder": choice_order, 
          "SubSelector": "TX", 
          "DataExportTag": data_export_tag, 
          "Choices": choices,
          "Configuration": {
            "QuestionDescriptionOption": "UseText"
          }, 
          "ChoiceGroups": {
            "cg_2": {
              "Randomization": {
                "Type": "All"
              }, 
              "GroupLabel": "Terms", 
              "Options": {
                "HideTitle": False, 
                "Selection": "SAWithinGroup"
              }, 
              "ChoiceGroupOrder": choice_order_intrusion
            }, 
            "cg_1": {
              "Randomization": {
                "Type": "None"
              }, 
              "GroupLabel": "Answer Confidence", 
              "Options": {
                "HideTitle": False, 
                "Selection": "SAWithinGroup"
              }, 
              "ChoiceGroupOrder": choice_order_confidence
            }
          }, 
          "ChoiceGroupOrder": [
            "cg_2", 
            "cg_1"
          ]
        }, 
        "PrimaryAttribute": f'QID{question_id}'
    }
    
    if include_confidence:
        return intruder_question_confidence
    
    return intruder_question


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--file", dest="file", help="File path for input files", required=True)
    arg_parser.add_argument("-o", "--output", dest="output", help="Output file name", required=True)
    arg_parser.add_argument("-s", "--survey_name", dest="survey_name", help="Survey name", required=True)
    arg_parser.add_argument("-n", "--num_intruders", dest="num_intruders",
                            type=int, help="Number of Topics to generate intruder experiment for",
                            default=20)
    arg_parser.add_argument("-m", "--model_name", dest="model_name", help="Useful model name param", default="topics")
    arg_parser.add_argument("--num_terms", dest="num_terms", type=int, help="Number of top terms to show", default=5)
    arg_parser.add_argument("--sample_top_terms", dest="sample_top_terms", action="store_true", default=False, help="Sample the topic terms from top 10 instead of taking top 5")

    arg_parser.add_argument("--include_confidence", dest="include_confidence", action="store_true", default=False, help="Include an option specify confidence")
    arg_parser.add_argument("--multi", dest="multi", help="Multi topic file experiment", default=None)
    arg_parser.add_argument("--file_2", dest="file_2", help="Second topic file")
    arg_parser.add_argument("--model_name_2", dest="model_name_2", help="Useful model name param for the second topic file")
    arg_parser.add_argument("--num_rand_questions", default="25", help="Number of questions to show out of the max")
    arg_parser.add_argument("--prolific", help="URL of the link back to Prolific")
    arg_parser.add_argument("--file_3", dest="file_3", help="Third topic file")
    arg_parser.add_argument("--model_name_3", dest="model_name_3", help="Useful model name param for the third topic file")

    args = arg_parser.parse_args()

    if args.prolific:
        survey_template = json.load(open("Intrusion_Template_Prolific.qsf"))
    else:
        survey_template = json.load(open("Intrusion_Template.qsf"))

    questions = []
    # THIS NEED TO START AT 3 TO TAKE INTO ACCOUNT THE 2 INTRO BLOCKS
    question_id = 3

    # THIS IS NEEDED TO ACCOUNT FOR THE ADDITIONAL PROLIFIC ID QUESTION
    if args.prolific:
        question_id =  4

    if args.multi:
      
        multi_topics_1 = load_multi_topics_file(args.file, args.model_name)[args.multi]
        multi_topics_2 = load_multi_topics_file(args.file_2, args.model_name_2)[args.multi]
        multi_topics_3 = load_multi_topics_file(args.file_3, args.model_name_3)[args.multi]

        intruder_setup_1 = setup_word_intrusion(
          multi_topics_1,
          n=args.num_intruders,
          topic_idxs=None,
          num_terms=args.num_terms,
          sample_top_topic_terms=args.sample_top_terms
        )

        json.dump(intruder_setup_1,
            open(f"{args.output}_{args.model_name}.json", "w+"),
            indent=2)

        intruder_setup_2 = setup_word_intrusion(
          multi_topics_2,
          n=args.num_intruders,
          topic_idxs=None,
          num_terms=args.num_terms,
          sample_top_topic_terms=args.sample_top_terms
        )

        json.dump(intruder_setup_2,
            open(f"{args.output}_{args.model_name_2}.json", "w+"),
            indent=2)

        intruder_setup_3 = setup_word_intrusion(
          multi_topics_3,
          n=args.num_intruders,
          topic_idxs=None,
          num_terms=args.num_terms,
          sample_top_topic_terms=args.sample_top_terms
        )

        json.dump(intruder_setup_3,
            open(f"{args.output}_{args.model_name_3}.json", "w+"),
            indent=2)

        questions = []
                
        for intruder in intruder_setup_1:
            full_terms = [term for term in intruder["topic_terms"]]
            full_terms.append(intruder["intruder_term"])

            questions.append(
                format_intruder_question(
                    question_id=question_id,
                    topic_id=intruder["topic_id"],
                    topic_intruder_id=intruder["intruder_id"],
                    terms=full_terms,
                    model_name=args.model_name,
                    include_confidence=args.include_confidence)
            )
            question_id += 1

        for intruder in intruder_setup_2:
            full_terms = [term for term in intruder["topic_terms"]]
            full_terms.append(intruder["intruder_term"])

            questions.append(
                format_intruder_question(
                    question_id=question_id,
                    topic_id=intruder["topic_id"],
                    topic_intruder_id=intruder["intruder_id"],
                    terms=full_terms,
                    model_name=args.model_name_2,
                    include_confidence=args.include_confidence)
            )
            question_id += 1

        for intruder in intruder_setup_3:
            full_terms = [term for term in intruder["topic_terms"]]
            full_terms.append(intruder["intruder_term"])

            questions.append(
                format_intruder_question(
                    question_id=question_id,
                    topic_id=intruder["topic_id"],
                    topic_intruder_id=intruder["intruder_id"],
                    terms=full_terms,
                    model_name=args.model_name_3,
                    include_confidence=args.include_confidence)
            )
            question_id += 1

    else:
        topics = load_topics_file(args.file)
        intruder_setup = setup_word_intrusion(
            topics,
            n=args.num_intruders,
            topic_idxs=None,
            num_terms=args.num_terms,
            sample_top_topic_terms=args.sample_top_terms)

        json.dump(intruder_setup,
            open(f"{args.output}_{args.model_name}.json", "w+"),
            indent=2)
        

        for intruder in intruder_setup:
            # This ensures the intruder is always the last work in the list and will be the last encoded term
            full_terms = [term for term in intruder["topic_terms"]]
            full_terms.append(intruder["intruder_term"])

            questions.append(
                format_intruder_question(
                    question_id=question_id,
                    topic_id=intruder["topic_id"],
                    topic_intruder_id=intruder["intruder_id"],
                    terms=full_terms,
                    model_name=args.model_name,
                    include_confidence=args.include_confidence)
            )
            question_id += 1

    print(f"Generating {len(questions)} questions")

    # Step 1: Find and tweak the Survey Blocks sections
    if survey_template["SurveyElements"][0]["PrimaryAttribute"] == "Survey Blocks":
        update_survey_blocks_element(
            survey_template["SurveyElements"][0], 
            questions,
            num_random=args.num_rand_questions
        )
        #survey_template["SurveyElements"][0] = format_survey_blocks(questions, args.num_rand_questions)
        #for payload in survey_template["SurveyElements"][0]["Payload"]:
        #    if payload["ID"] == "BL_2soHZqi6foVxie1":
        #        payload["Options"]["Randomization"]["Advanced"]["TotalRandSubset"] = str(args.num_rand_questions)
                
        
    # Step 2: Wipe the existing old questions that are there for easy reference
    idx_to_delete = []
    for idx, element in enumerate(survey_template["SurveyElements"]):

        if element["Element"] == "SQ":
            if element.get("Payload", {}).get('QuestionType') == "MC":
                idx_to_delete.append(idx)

    survey_template["SurveyElements"] = [element for idx, element in enumerate(survey_template["SurveyElements"])
                                        if not idx in idx_to_delete]
        
    if args.prolific:
        set_redirect_url(survey_template["SurveyElements"], args.prolific )

    # Step 3: Append the new questions to the end
    for question in questions:
        survey_template["SurveyElements"].append(question)

    day = datetime.date.today().strftime('%Y-%m-%d')
    survey_template["SurveyEntry"]["SurveyName"] = f"Intrusion {args.survey_name.title()} {day}"
        
    if args.multi == "nytimes":
        set_nytimes_dataset(survey_template["SurveyElements"])
    if args.multi == "wikitext":
        set_wikitext_dataset(survey_template["SurveyElements"])

    json.dump(survey_template,
            open(f"{args.output}.qsf", "w+"),
            indent=2)



    

