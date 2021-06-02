

import argparse
import json
import requests

API_KEY = open("qualtrics_key").read().replace("\n", "")

ENDPOINT = "https://iad1.qualtrics.com/API/v3"


users = [
    "UR_56juIFbMBWLU9jo",
    "UR_bIMXWqoIF8GNR7E",
    "UR_4O2EvhVhx5UVoQS"
]

# may 17
survey_ids = [
    "SV_ezlYrHXuI4oxV8W",
    "SV_0Va3pTzbIGiS8tM",
    "SV_50zRZfCRm5wvvxA",
    "SV_80UG9Vuo5KHKbFI",
    "SV_3z2XeepZD1uheui",
    "SV_6GuDFnyQX0IhAeq",
    "SV_4Ob1KEJ0trA5KCy",
    "SV_1NyEUJ5LX8KtHFQ",
    "SV_3z661hKZxFMnewC",
    "SV_bqpnsYfAAixidLM"
]

# may 22
survey_ids = [
    "SV_3C7Uiny38kWgS8u",
    "SV_d05P9MJ18BPKJzU",
    "SV_42D15fE78lMVZgq",
    "SV_4TpNIilU2NiT0Gi",
    "SV_3Cnx3iEAyukf0Dc",
    "SV_bsgYZgUnyITeVDM",
    "SV_1SS889w7P5jPCho",
    "SV_e4YDrmZt5H3Ek1E",
    "SV_50dy6i7mkv7yO34",
    "SV_8A19cs0dtknYHHw"
]

for survey_id in survey_ids:
    for user_id in users:

        resp = requests.post(
            f"https://ca1.qualtrics.com/API/v3/surveys/{survey_id}/permissions/collaborations", 
            headers={"X-API-TOKEN": API_KEY},
            json={
                "recipientId": user_id,
                "permissions": {
                    "surveyDefinitionManipulation": {
                        "copySurveyQuestions": True,
                        "editSurveyFlow": True,
                        "useBlocks": True,
                        "useSkipLogic": True,
                        "useConjoint": True,
                        "useTriggers": True,
                        "useQuotas": True,
                        "setSurveyOptions": True,
                        "editQuestions": True,
                        "deleteSurveyQuestions": True,
                        "useTableOfContents": True,
                        "useAdvancedQuotas": True
                    },
                    "surveyManagement": {
                        "editSurveys": True,
                        "activateSurveys": True,
                        "deactivateSurveys": True,
                        "copySurveys": True,
                        "distributeSurveys": True,
                        "deleteSurveys": True,
                        "translateSurveys": True
                    },
                    "response": {
                        "editSurveyResponses": True,
                        "createResponseSets": True,
                        "viewResponseId": True,
                        "useCrossTabs": True,
                        "useScreenouts": True
                    },
                    "result": {
                        "downloadSurveyResults": True,
                        "viewSurveyResults": True,
                        "filterSurveyResults": True,
                        "viewPersonalData": True
                    }
                }
            })

        print(resp.status_code)
        if resp.status_code > 299:
            print(resp.__dict__)