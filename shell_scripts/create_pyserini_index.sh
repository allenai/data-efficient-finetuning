
P3_DATA=$1
INDEX_OUTPUT=$2
THREADS=$3

# of course, you'll have to install pyserini for this to work first :)
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $P3_DATA \
  --index $INDEX_OUTPUT \
  --generator DefaultLuceneDocumentGenerator \
  --threads $THREADS \
  --storePositions --storeDocvectors