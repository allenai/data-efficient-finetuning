
QUERY_LOCATION=queries  # should contain tsvs of name {dataset}.tsv with format "id\tquery_string" (id can be whatever)
P3_DATA=$1  # location of jsonl where each line is a p3 instance and the index doc ids == the index of datapoints in this file.

# searching
for dataset in rte anli_r1 anli_r2 anli_r3 casehold cb copa drop hellaswag qasper storycloze wic wsc winogrande
do
    python -m pyserini.search.lucene \
        --index pyserini_index \
        --topics $QUERY_LOCATION/${dataset}.tsv \
        --output ${dataset}_query.txt \
        --bm25 \
        --threads 64 \
        --hits 500
    # sort and uniq as the pyserini stuff will result in dups.
    cut -f3 -d ' ' ${dataset}_query.txt | sort | uniq > $QUERY_LOCATION/${dataset}_idxes.txt
    rm ${dataset}_query.txt # remove this line to keep pyserini output
    echo "${dataset} done processing!"
done

echo "Indexes retrieved. Writing out data files..."

# we run indices to instances script now since it takes ~10min to run through the file once, so doing one pass with
# all indices is much faster than running through #datasets times.
python scripts/indices_to_file.py \
    --infile_path $QUERY_LOCATION \
    --outfile_path $QUERY_LOCATION \
    --suffix bm25_500hits.jsonl \
    --p3_data $P3_DATA

echo "Data files written to $QUERY_LOCATION!"
