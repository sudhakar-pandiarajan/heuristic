# Query-by-Example Spoken Term Detection for Zero-Resource Languages Using Heuristic Pattern Match

## Overview
Python implementation of Query-by-Example Spoken Term Detection Using Heuristic Search. In our approach, the STD task was accomplished in two stages: 
1. Speaker Independent Spoken Content Representation
2. Spoken Term Detection using heuristic pattern match

## Speaker Independent Spoken Content Representation
In this stage, the acoustic feature representation of the speech signal was converted to a single target speaker using our voice conversion model. 
![Speaker Style Conversion](https://github.com/sudhakar-pandiarajan/heuristic/blob/main/speaker_style_convert.png)
In this demo, the spoken content of the speaker 1, 2, and 3 is converted to speaker 2. Further, the similarity realisation before and after conversion was verfied using the correlation values. (see below plot)
![Verification](https://github.com/sudhakar-pandiarajan/heuristic/blob/main/speaker_style_verify.png)

## Heuristic similarity pattern detection
In this stage, the heuristic cost based similarity measure was used to detect the spoken query content in the document. The below plot demonstrate the similarity matrix and the heuristic similarity exhibited between a spoken query and document. The peak values in the heuristic similarity indicates the region of similarity. The lines in red color represents the ground truth region.
![Heuristic Similarity](https://github.com/sudhakar-pandiarajan/heuristic/blob/main/heuristic_similarity_match.png)

## Disclaimer
Use code at own risk.

## Datasets 
[Microsoft Speech Corpus for low-resource Indian languages](https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e)

## Python package dependencies
1.  glob
2.  numpy
3.  pandas
4.  tqdm
5.  pickle
6.  shennong
7.  librosa
8.  sklearn
9.  torch
10. torchaudio
11. scipy


## Reference
```
@article{sudhakarqbe2023,
author = {P, Sudhakar and K, Sreenivasa Rao and Mitra, Pabitra},
title = {Query-by-Example Spoken Term Detection for Zero-Resource Languages Using Heuristic Search},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2375-4699},
url = {https://doi.org/10.1145/3609505},
doi = {10.1145/3609505},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = {jul}
}
