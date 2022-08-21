# Query-by-Example Spoken Term Detection for Zero-Resource Languages Using Heuristic Search

## Overview
Python implementation of Query-by-Example Spoken Term Detection Using Heuristic Search. In our approach, the STD task was accomplished in two stages. 
1. Speaker Independent Spoken Content Representation
2. Spoken Term Detection 

## Speaker Independent Spoken Content Representation
In this stage, the acoustic feature representation of the speech signal was converted to a single target speaker using our voice conversion model. 
![alt text](https://github.com/sudhakar-pandiarajan/heuristic/speaker_style_convert.png)
In this demo, the spoken content of the speaker 1, 2, and 3 were converted to the speaker 2. Further, the similarity realisation before and after conversion was verfied using the plot below
![alt text](https://github.com/sudhakar-pandiarajan/heuristic/speaker_style_verify.png)

## Heuristic Similarity detection
In this stage, the heuristic cost based similarity measure was used to detect the  spoken query content in the document. The below plot demonstrate the similarity and heuristic similarity significance the region
![alt text](https://github.com/sudhakar-pandiarajan/heuristic/heuristic_similarity_match.png)

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
9.  itertools
10. spy
11. som
12. scipy


## Reference
```
@article{sudhakar2022ake,
  title={Query-by-Example Spoken Term Detection for Zero-Resource Languages Using Heuristic Search},
  author={Sudhakar P, Sreenivasa Rao K, and Pabitra Mitra},
  booktitle={ACM Trans. Asian Low-Resour. Lang. Inf. Process},
  year={2022},
  pages={xx-xx},
}
