## Keyword extraction models for Russian

### Datasets

from https://github.com/mannefedov/ru_kw_eval_datasets

habr -- HabraHabr https://habr.com/

ng -- Независимая Газета http://www.ng.ru/

rt -- Russia Today https://russian.rt.com/

cl -- Cyberleninka https://cyberleninka.ru/

### Preprocessing options (nltk, pymorphy):
  1) tokenization  
  2) lemmatization
  3) extracting nouns and adjectives in nominative case: "ясная ночь", not "ясный ночь"
  4) ngrams spliteration (n=2)

_Further improvements: implement udpipe for tokenization and lemmatization, combine with pymorphy to extract_

### Approaches

1. Simple TFIDF method
2. SCAKE graph method https://arxiv.org/pdf/1811.10831v1.pdf
3. NN approach (in progress)


### Training and evaluation mode

```
usage: model_trainer.py [-h] [-d [{all,rt,ng,habr,cl}]] [-o OUTPUT_PATH]
                        [-m [{tfidf,scake}]]

Keyword extractor

optional arguments:
  -h, --help            show this help message and exit
  -d [{all,rt,ng,habr,cl}], --dataset [{all,rt,ng,habr,cl}]
  -o OUTPUT_PATH, --output OUTPUT_PATH
  -m [{tfidf,scake}], --model [{tfidf,scake}]
```

### Results on NG dataset

#### TFIDF

| Metric | Value |
| -------- | ------|
|Precision | 0.1385|
|Recall |  0.2649|
|F1 |  0.1733|
|Jaccard |  0.1014|

#### SCAKE
(much slower than tfidf)

| Metric | Value |
| -------- | ------|
|Precision |  |
|Recall |  |
|F1 |  |
|Jaccard |  |

#### NN

Model implemented but does not perform well. Further investigation needed

### Implementation mode

File "**main.py**" contains text that is normalized and then kws are extracted using different approaches in parallel with the usage of celery
