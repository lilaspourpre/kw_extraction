## Keyword extraction models for Russian

### Datasets



### Preprocessing options (nltk, pymorphy):
  1) tokenization  
  2) lemmatization
  3) extracting nouns and adjectives in nominative case: "ясная ночь", not "ясный ночь"
  4) ngrams spliteration (n=2)

_Further improvements: implement udpipe for tokenization and lemmatization, combine with pymorphy to extract_

### Approaches

1. Simple TFIDF method
2. SCAKE graph method
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

### Implementation mode


