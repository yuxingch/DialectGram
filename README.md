# DialectGram: Detection of Dialectal Changes with Multi-geographic Resolution Analysis

![heat-map-gas](./image/gas.png?raw=true "Word heat map of gas")

## Introduction
Several computational models have been developed to detect and analyze dialect variation in recent years. Most of these models assume a predefined set of geographical regions over which they detect and analyze dialectal variation. However, dialect variation occurs at multiple levels of geographic resolution ranging from cities within a state, states within a country, and between countries across continents. In this work, we propose a model that enables detection of dialectal variation at multiple levels of geographic resolution obviating the need for a-priori definition of the resolution level. Our method **DialectGram**, learns dialect-sensitive word embeddings while being agnostic of the geographic resolution. Specifically it only requires one-time training and enables analysis of dialectal variation at a chosen resolution post-hoc -- a significant departure from prior models which need to be re-trained whenever the pre-defined set of regions changes. Furthermore, **DialectGram** explicitly models senses thus enabling one to estimate the proportion of each sense usage in any given region. Finally, we quantitatively evaluate our model against other baselines on a new evaluation dataset *DialectSim* (in English) and show that **DialectGram** can effectively model linguistic variation.

## Visualization Demo
You can visualize our word maps here: [demo](https://yuxingch.github.io/DialectGram/demo/main.html)

## Dataset
### Geo-Tweets2019
The new English Twitter corpus (Geo-Tweets2019) used in this project can be found at [Geo-Twitter2019](https://github.com/hjian42/Geo-Twitter2019), which is built for training dialect-sensitive word embeddings.

### DialectSim
A new validation set for evaluating the quality of English region-specific word embeddings between the UK and the USA (i.e. at the country level). Can be found at `./data/DialectSim_{train,test}.csv` in this repository.

## Models
We include the baseline models (`Frequency`/`Syntactic`/`GEODIST`) and `DialectGram` model that are used in our project.
### Baseline Models
  - `Frequency` Model
  - `Syntactic` Model
  - `GEODIST` Model: a model to learn region-specific word embeddings using the Skip-gram framework. We implemented this model in PyTorch, based on the approach presented in [Freshman or Fresher? Quantifying the Geographic Variation of Language in Online Social Media](https://arxiv.org/pdf/1510.06786.pdf) and the [Java code](https://github.com/dbamman/geoSGLM) for learning geographically-informed word embeddings. 

### DialectGram Model 
A novel method to learn dialect-sensitive word embeddings from region-agnostic data, based on AdaGram \[[1](https://github.com/sbos/AdaGram.jl), [2](https://github.com/lopuhin/python-adagram)\].

## Requirements
### Baseline Models
- cycler            0.10.0 
- folium            0.10.0
- joblib            0.13.2 
- kiwisolver        1.1.0  
- matplotlib        3.1.0  
- numpy             1.16.3 
- pandas            0.24.2 
- pep8              1.7.1
- plotly            4.1.1  
- Pillow            6.0.0  
- pip               19.1.1 
- pyparsing         2.4.0  
- python-dateutil   2.8.0  
- pytz              2019.1
- reverse_geocoder  1.5.1
- scikit-learn    0.21.2 
- scipy           1.3.0  
- setuptools      41.0.1 
- six             1.12.0 
- torch           1.1.0  
- torchvision     0.3.0  
- tqdm            4.32.1 
- wheel           0.33.4

### DialectGram
- [Julia](https://github.com/JuliaLang/julia) 0.4
- To run `DialectGram` model, we need to install Python-AdaGram package from source:
  ```
  $ pip install Cython numpy
  $ pip install git+https://github.com/lopuhin/python-adagram.git
  ```
  , and install AdaGram.jl as well:
  ```
  Pkg.clone("https://github.com/sbos/AdaGram.jl.git")
  Pkg.build("AdaGram")
  ```

## Sample Usage
### Baseline Models
For `Frequency` and `Syntactic` models, we can build and directly evaluate them using `{freq,synt}_eval.py`:
```
# Frequency model
python freq_eval.py

# Syntactic model
python synt_eval.py
```
If we want to investigate a specific word, run:
```
# Frequency model
python frequencies.py "word_we_want_to_test"

# Syntactic model
python syntactic.py "word_we_want_to_test"
```

#### Training `GEODIST`
```
# using default parameters
python geodist_run.py

# or, we can specify the values of the following parameters:
python geodist_run.py --batch=128 --window=10 --freq=20 --step=80000 --dim=100 --lr=0.05 --dir='./outputs'
```
#### Evaluating `GEODIST`
```
python geodist_eval.py
```

### `DialectGram`


## Citation
Jiang, Hang*; Haoshen Hong*; Yuxing Chen*; and Vivek Kulkarni. 2019. [DialectGram: Automatic Detection of Dialectal Changes with Multi-geographic Resolution Analysis](https://arxiv.org/abs/1910.01818). To appear in *Proceedings of the Society for Computation in Linguistics*. New Orleans: Linguistic Society of America. 

```
@inproceedings{Jiang:Hong:Chen:Kulkarni:2020:SCiL,
  Author = {Jiang, Hang  and  Hong, Haoshen  and  Chen, Yuxing  and  Kulkarni, Vivek},
  Title = {DialectGram: Detection of Dialectal Changes with Multi-geographic Resolution Analysis},
  Booktitle = {Proceedings of the Society for Computation in Linguistics},
  Location = {New Orleans},
  Publisher = {Linguistic Society of America},
  Address = {Washington, D.C.},
  Year = {2020}}
```
