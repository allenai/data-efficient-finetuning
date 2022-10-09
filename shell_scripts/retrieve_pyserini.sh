
# searching
for dataset in rte anli_r1 anli_r2 anli_r3 casehold cb copa drop hellaswag qasper storycloze wic wsc winogrande
do
    python -m pyserini.search.lucene \
        --index pyserini_index \
        --topics queries/${dataset}.tsv \
        --output ${dataset}_query.txt \
        --bm25 \
        --threads 64 \
        --hits 500
    # sort and uniq as the pyserini stuff will result in dups.
    cut -f3 -d ' ' ${dataset}_query.txt | sort | uniq > queries/${dataset}_idxes.txt
    rm ${dataset}_query.txt
    echo "${dataset} done processing!"
    # note that you have to run the indices to instances script. I leave it out here since running it in batch is much faster.
done
