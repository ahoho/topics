python download_and_split.py --data-dir raw --output-dir intermediate/full --val-split 0.0 --test-split 0.0 --rejoin-split-terms --use-raw

OUTPUT_DIR=neurips_reference
soup-nuts preprocess \
    intermediate/full/train-raw.txt \
    processed/${OUTPUT_DIR} \
    --test-path raw/full-dump-2017/raw_articles.txt \
    --input-format text \
    --output-text \
    --lines-are-documents \
    --lowercase \
    --max-doc-size 19000 \
    --min-doc-size 5 \
    --min-doc-freq 31 \
    --max-doc-freq 0.9 \
    --n-process 12 \
    --detect-entities \
    --token-regex wordlike \
    --no-double-count-phrases \
    --min-chars 2

# Rename "test" to "full"
mv processed/${OUTPUT_DIR}/test.ids.json processed/${OUTPUT_DIR}/full.ids.json
mv processed/${OUTPUT_DIR}/test.dtm.npz processed/${OUTPUT_DIR}/full.dtm.npz
mv processed/${OUTPUT_DIR}/test.txt processed/${OUTPUT_DIR}/full.txt

# Create val and test splits
python filter_and_split.py processed/${OUTPUT_DIR} --val_split 4200