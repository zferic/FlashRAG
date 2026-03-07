#!/bin/bash

CORPUS_PATH=/projects/nucar/feric.z/FLashRAG-z/test_sample_2018_sent.jsonl
BASE_SAVE_DIR=/projects/nucar/feric.z/wikiindex2

FAISS_TYPES=(
    "Flat"
    "SQfp16"
    "SQ8"
    "SQ4"
    "IVF300,Flat"
    "IVF1000,Flat"
    "IVF300,SQfp16"
    "IVF300,SQ8"
    "IVF300,SQ4"
    "IVF1000,SQfp16"
    "IVF1000,SQ8"
    "IVF1000,SQ4"
)
MAX_LENGTHS=(128 256)

for faiss_type in "${FAISS_TYPES[@]}"; do
    for max_length in "${MAX_LENGTHS[@]}"; do
        save_dir="${BASE_SAVE_DIR}/index_${faiss_type//,/-}_ml${max_length}"
        echo "Running: faiss_type=${faiss_type} max_length=${max_length}"
        CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
            --retrieval_method e5 \
            --model_path intfloat/e5-small-v2 \
            --corpus_path $CORPUS_PATH \
            --save_dir $save_dir \
            --use_fp16 \
            --max_length $max_length \
            --batch_size 512 \
            --pooling_method mean \
            --faiss_type "$faiss_type" \
            --save_embedding
    done
done
