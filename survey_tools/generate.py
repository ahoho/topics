
import argparse
import json
import requests

API_KEY = open("qualtrics_key").read().replace("\n", "")

ENDPOINT = "https://iad1.qualtrics.com/API/v3"


# Header parameter name: X-API-TOKEN
#resp = requests.get(f"{ENDPOINT}/surveys", headers={"X-API-TOKEN": API_KEY})
#print(resp, resp.content)


def api(path, method="GET"):
    if method == "GET":
        response = requests.get(f"{ENDPOINT}/{endpoint}", headers={"X-API-TOKEN": API_KEY})
    if method == "POST":
        response = requests.post(f"{ENDPOINT}/{endpoint}", headers={"X-API-TOKEN": API_KEY})

    if response.status_code <= 299:
        return response.json()
    else:
        print("Error: ", response.content)
    return None


"""
files = [
    "april_2021_surveys/yelp_ratings_neural.qsf",
    "april_2021_surveys/yelp_intrusion_neural.qsf",
    "april_2021_surveys/yelp_intrusion_10_terms.qsf",
    "april_2021_surveys/legislation_intrusion.qsf"
]
"""
#"../may_5_2021/bbc_intrusion.qsf"
#"../may_5_2021/bbc_ratings.qsf"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--filepath", dest="filepath", help="File path for input files", required=True)

    args = arg_parser.parse_args()
    files = [
        args.filepath
    ]
    for file_ in files:
        #print(file_)
        json_info = json.load(open(file_))
        #name = file_.split("/")[-1].title()
        entry = json_info["SurveyEntry"]
        #entry["SurveyName"] = name
        name = entry["SurveyName"]
        print(name)
        resp = requests.post(
            "https://sjc1.qualtrics.com/API/v3/survey-definitions",
            headers={"X-API-TOKEN": API_KEY},
            json={
                "SurveyEntry": entry,
                "SurveyElements": json_info["SurveyElements"]
            }
        )

        print(resp.status_code)#, resp.json())
        payload = resp.json()
        #print(payload)

        survey_id = payload["result"]["SurveyID"]
        print("================")
        resp = requests.post(f"https://sjc1.qualtrics.com/API/v3/survey-definitions/{survey_id}/versions",
            headers={"X-API-TOKEN": API_KEY},
            json={
                "Published": True,
                "Description": "Topic model evaluation survey"
            }
        )
        #print(resp.__dict__)
        print(resp.status_code)#, resp.json())

        resp = requests.put(f"https://sjc1.qualtrics.com/API/v3/surveys/{survey_id}",
            headers={"X-API-TOKEN": API_KEY},
            json={    
                "name": name,
                "isActive": True,
                "expiration": {
                    "startDate": "2021-05-20T02:54:03Z",
                    "endDate":"2022-05-20T02:54:03Z"
                },
                "ownerId": "UR_1MmbX0D3bAle6sR"
            })
        #print(resp.__dict__)
        print(resp.status_code, resp.json())
        print(f"https://umdsurvey.umd.edu/jfe/form/{survey_id}")
        print("************************************")
        """
        print("Survey ID: ", survey_id)
        resp = requests.post("https://sjc1.qualtrics.com/API/v3/distributions",
            headers={"X-API-TOKEN": API_KEY},
            json={
                "surveyId": survey_id,
                "linkType": "Individual",
                "description": "distribution 2021-05-11 00:00:00",
                "action": "CreateDistribution",
                "expirationDate": "2022-05-11 00:00:00",
                "mailingListId": "CG_6F1gRt186CZOVoh"
            }
        )
        print(resp.__dict__)
        """