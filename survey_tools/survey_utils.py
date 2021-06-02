import json

def load_multi_topics_file(filepath, model_type):
    topic_file = json.load(open(filepath))
    # first set of keys are the datasets
    datasets = {
      key: [
              {
                  "topic_id": idx,
                  "terms": topic,
                  "model_name": key,
                  "model_type": model_type
              }
              for idx, topic in enumerate(topic_file[key]["topics"])
      ]
      for key in topic_file.keys()
    }
    return datasets


def load_topics_file(filepath, delimiter=","):
    reader = open(filepath)
    # This is risky 'bad' logic for dynamically determining the delimiter
    lines = [line.replace("\t", " ") for line in reader.readlines()]
    if len(lines[0].split(" ")) > len(lines[0].split(",")):
        delimiter = " "
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

def set_nytimes_dataset(survey_elements, qid="QID2"):
    update_dataset_blurb(
        survey_elements,
        dataset="The New York Times",
        blurb="The New York Times is an American newspaper featuring articles from 1987 to 2007. Sections from a typical paper include <em>International</em>, <em>National</em>, <em>New York Regional</em>, <em>Business</em>, <em>Technology</em>, and <em>Sports</em> news; features on topics such as <em>Dining</em>, <em>Movies</em>, <em>Travel</em>, and <em>Fashion</em>; there are also obituaries and opinion pieces. <br><br>This study should take approximately 10-15 minutes to complete. <br><br>Your response will be completely anonymous.",
        qid=qid
    )

def set_wikitext_dataset(survey_elements, qid="QID2"):
    update_dataset_blurb(
        survey_elements,
        dataset="Wikipedia",
        blurb='Wikipedia is an online encyclopedia covering a huge range of topics. Articles can include biographies ("George Washington"), scientific phenomena ("Solar Eclipse"), art pieces ("La Danse"), music ("Amazing Grace"), transportation ("U.S. Route 131"), sports ("1952 winter olympics"), historical events or periods ("Tang Dynasty"), media and pop culture ("The Simpsons Movie"), places ("Yosemite National Park"), plants and animals ("koala"), and warfare ("USS Nevada (BB-36)"), among others. <br><br>This study should take approximately 10-15 minutes to complete. <br><br>Your response will be completely anonymous.',
        qid=qid
    )

def update_dataset_blurb(survey_elements, dataset, blurb, qid="QID2"):
    for element in survey_elements:
      # update the second section
        if element["PrimaryAttribute"] == qid:
            element["Payload"]["QuestionText"] += f"<br> In this survey, the word lists are based on a computer analysis of {dataset}. <br><br>" + blurb


def set_redirect_url(survey_elements, redirect_url):
    for element in survey_elements:
        if element["PrimaryAttribute"] == "Survey Options":
            element["Payload"]["EOSRedirectURL"] = redirect_url
        

def update_survey_blocks_element(survey_blocks_element, questions, num_random=25):
    survey_blocks_element["Payload"][-1]["BlockElements"] = [
        {"Type": "Question", "QuestionID": question["Payload"]["QuestionID"]} for question in questions]
    survey_blocks_element["Payload"][-1]["Options"]["RandomizeQuestions"] = "RandomWithOnlyX"
    survey_blocks_element["Payload"][-1]["Options"]["Randomization"]["Advanced"]["TotalRandSubset"] = num_random

def format_survey_blocks(questions, n=25):
    """
    The SurveyBlocks section has to be extended with the set of questions
    """
    
    return {
      "SurveyID": "SV_5sXmuibskKlpHmJ",
      "Element": "BL",
      "PrimaryAttribute": "Survey Blocks",
      "SecondaryAttribute": None,
      "TertiaryAttribute": None,
      "Payload": [
        {
          "Type": "Default",
          "Description": "Introduction Block",
          "ID": "BL_cvxoPps3HflxlsN",
          "BlockElements": [
            {
              "Type": "Question",
              "QuestionID": "QID1"
            },
            {
              "Type": "Question",
              "QuestionID": "QID2"
            }
          ]
        },
        {
          "Type": "Trash",
          "Description": "Trash / Unused Questions",
          "ID": "BL_0IH9ow1MoCfyMW9"
        },
        {
          "Type": "Standard",
          "SubType": "",
          "Description": "Intruder Experiment",
          "ID": "BL_2soHZqi6foVxie1",
          "BlockElements": [{"Type": "Question", "QuestionID": question["Payload"]["QuestionID"]} for question in questions],
          "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "RandomWithOnlyX",
            "Randomization": {
              "Advanced": {
                "QuestionsPerPage": 0,
                "TotalRandSubset": n
              }
            }
          }
        }
      ]
    }


def load_fake_topics():
    fake_topics = json.load(open("fake_topics.json"))
    idx = 10000
    topics = []
    for key in fake_topics:
        
        for fake_source in ["in_vocab"]:
            for terms in fake_topics[key][fake_source]:
                topics.append({
                    "topic_id": idx,
                    "terms": terms,
                    "model_name": key +"_fake",
                    "model_type": fake_source
                })
                idx += 1
    return topics
      

      