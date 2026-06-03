#!/bin/bash
pip install gdown

gdown "1v4MaKYwvCowNQCXZSvHOBA6_mViCvjlQ" -O /workspace/tools/monaco_preprocessing/data/restructured_monaco.jsonl
gdown --folder 'https://drive.google.com/drive/folders/1oearPMW-Lk5sPXWj-7UKPaIVlcio_7Wq' -O /workspace/tools/monaco_preprocessing/data/scraped_wiki_singleurl_olderrev_full
gdown --folder 'https://drive.google.com/drive/folders/1x50pkquWOaOgUpQfl_5aQ7tztoqD5RIk' -O /workspace/tools/monaco_preprocessing/data/monaco_wiki_vec_db