#!/bin/sh


echo "\nThis script will standardize formatting, sample data, and update the coherence file \n"
echo "Reformatting original data"


python ProcessRatingsDatasets.py 'all_data' 'nyt'
python ProcessRatingsDatasets.py 'all_data' 'wikitext'
python ProcessIntrusionDatasets.py 'all_data' 'nyt'
python ProcessIntrusionDatasets.py 'all_data' 'wikitext'

echo "Done \n"
echo "Sampling 15 annotations per topic.  Need to update ratings task with gap"

python SampleData.py 'all_data' 'nytimes' 'intrusion'
python SampleData.py 'all_data' 'wikitext' 'intrusion'
python SampleData.py 'all_data' 'wikitext' 'ratings'
python SampleData.py 'all_data' 'nytimes' 'ratings'

echo "Done \n"
echo "Updating the automated metrics file with raw and average counts"


python CreateJointFile.py 'all_data' 'nytimes' 'intrusion'
python CreateJointFile.py 'all_data' 'wikitext' 'intrusion'
python CreateJointFile.py 'all_data' 'nytimes' 'ratings'
python CreateJointFile.py 'all_data' 'wikitext' 'ratings'

echo "Done \n  File should now contain new keys"

python create_df_from_json.py 'all_data' 'all_data.json'
