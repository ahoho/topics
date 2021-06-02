

import argparse
import json
import requests
import time

API_KEY = open("qualtrics_key").read().replace("\n", "")

ENDPOINT = "https://az1.qualtrics.com/API/v3"

survey_ids = {
    "SV_50zRZfCRm5wvvxA": '../may_20_results/NYT_Ratings.csv',
    "SV_ezlYrHXuI4oxV8W": '../may_20_results/NYT_Intrusion_1.csv',
    "SV_0Va3pTzbIGiS8tM": '../may_20_results/Wiki_Intrusion_1.csv',
    "SV_80UG9Vuo5KHKbFI": '../may_20_results/Wiki_Ratings.csv',
    "SV_3z2XeepZD1uheui": '../may_20_results/NYT_Intrusion_2.csv',
    "SV_6GuDFnyQX0IhAeq": '../may_20_results/Wiki_Intrusion_2.csv'
}

survey_ids = {
    "SV_3C7Uiny38kWgS8u": "../may_22_2021/nyt_ratings.csv",
    "SV_d05P9MJ18BPKJzU": "../may_22_2021/wikitext_ratings.csv",
    "SV_42D15fE78lMVZgq":"../may_22_2021/nyt_intrusion_1.csv",
    "SV_4TpNIilU2NiT0Gi":"../may_22_2021/nyt_intrusion_2.csv",
    "SV_3Cnx3iEAyukf0Dc":"../may_22_2021/nyt_intrusion_3.csv",
    "SV_bsgYZgUnyITeVDM":"../may_22_2021/nyt_intrusion_4.csv",
    "SV_1SS889w7P5jPCho":"../may_22_2021/wikitext_intrusion_1.csv",
    "SV_e4YDrmZt5H3Ek1E":"../may_22_2021/wikitext_intrusion_2.csv",
    "SV_50dy6i7mkv7yO34":"../may_22_2021/wikitext_intrusion_3.csv",
    "SV_8A19cs0dtknYHHw":"../may_22_2021/wikitext_intrusion_4.csv"
}

survey_ids = {
    "SV_2tbdB9sXU8qdFjg": "../may_22_2021/gap_ratings.csv"
}



for survey_id in survey_ids:
    resp = requests.post(
        f"{ENDPOINT}/surveys/{survey_id}/export-responses",
        headers={"X-API-TOKEN": API_KEY},
        json={
            "format": "csv",
            "compress": False,
            "breakoutSets": False
        }
    )
    print(resp.__dict__)
    if resp.status_code < 299:
        progress_id = resp.json()["result"]["progressId"]
    else:
        raise Exception(json.dumps(resp.__dict__))

    # begin polling for the export progress
    file_id = None
    for i in range(0, 6):
        resp = requests.get(
            f"{ENDPOINT}/surveys/{survey_id}/export-responses/{progress_id}",
            headers={"X-API-TOKEN": API_KEY}
        )
        if resp.status_code > 299:
            raise Exception(json.dumps(resp.__dict__))
        resp_json = resp.json()
        if resp_json["result"]["status"] == "inProgress":
            print("Waiting...")
            time.sleep(5)
        if resp_json["result"]["status"] == "complete":
            file_id = resp_json["result"]["fileId"]
            break

    if not file_id:
        print("Exitting.... no file id.")
    
    print("Found file id!")
    resp = requests.get(
        f"{ENDPOINT}/surveys/{survey_id}/export-responses/{file_id}/file",
        headers={"X-API-TOKEN": API_KEY}
    )

    print(resp.status_code)
    writer = open(survey_ids[survey_id], "wb+")
    writer.write(resp.content)
    writer.close()