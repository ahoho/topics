python subsample_and_split.py --data-dir raw --output-dir intermediate/full --subsample-percentage 0.15 --min_doc_size 25

OUTPUT_DIR=neurips_reference
soup-nuts preprocess \
    intermediate/full/train.jsonl \
    processed/${OUTPUT_DIR} \
    --test-path intermediate/full/full.jsonl  \
    --input-format jsonl \
    --jsonl-text-key text \
    --jsonl-id-key id \
    --output-text \
    --lines-are-documents \
    --lowercase \
    --max-doc-size 5000 \
    --min-doc-size 5 \
    --min-doc-freq 85 \
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
python split.py processed/${OUTPUT_DIR} --val_split 42000