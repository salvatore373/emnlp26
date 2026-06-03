#!/bin/bash
pip install gdown

gdown 1fY5Zt3_RaplaRd3i4n6xMdoWRt5r4Z98 -O /workspace/tools/toolqa_preprocessing/data/toolqa_full.jsonl
gdown --folder 'https://drive.google.com/drive/folders/1OTFmf3n48GkgcRAxs2LefRrzReoHYjJh' -O /workspace/tools/toolqa_preprocessing/data/
mv /workspace/tools/toolqa_preprocessing/data/toolqa_files/* /workspace/tools/toolqa_preprocessing/data/
rm -rf /workspace/tools/toolqa_preprocessing/data/toolqa_files