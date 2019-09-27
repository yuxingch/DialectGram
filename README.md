# DialectGram: Automatic Detection of Dialectal Changes with Multi-geographic Resolution Analysis

![heat-map-gas](./image/gas.png?raw=true "Word heat map of gas")

## Introduction
We propose **DialectGram**, a method to detect dialectical variation across multiple geographic resolutions. In contrast to prior work, which requires apriori knowledge of the geographic resolution and the set of regions,  DialectGram automatically infers dialect-sensitive senses without these constraints using a nonparametric Bayesian extension of Skip-gram. Consequently, DialectGram only needs one-time training to enable an analysis of dialectical variation at multiple resolutions. To validate our approach, and establish a quantitative benchmark, we create a new corpus Geo-Tweets2019 with English tweets from the US and the UK, and new validation set DialectSim for evaluating word embeddings in American and British English. 

## Visualization Demo
[TODO]

## Dataset
### Geo-Tweets2019
The new English Twitter corpus (Geo-Tweets2019) used in this project can be found at [Geo-Twitter2019](https://github.com/hjian42/Geo-Twitter2019), which is built for training dialect-sensitive word embeddings.

### DialectSim
A new validation set for evaluating the quality of English region-specific word embeddings between the UK and the USA (i.e. at the country level). Can be found at `./data/DialectSim_{train,test}.csv` in this repository.

## Models
We include the baseline models (Frequency/Syntactic/GEODIST) and DialectGram model that are used in our project.
### Baseline Models
  - Frequency Model
  - Syntactic Model
  - GEODIST Model: a model to learn region-specific word embeddings using the Skip-gram framework. We implemented this model in PyTorch, based on the approach presented in [Freshman or Fresher? Quantifying the Geographic Variation of Language in Online Social Media](https://arxiv.org/pdf/1510.06786.pdf) and the [Java code](https://github.com/dbamman/geoSGLM) for learning geographically-informed word embeddings. 

### DialectGram Model 
A novel method to learn dialect-sensitive word embeddings from region-agnostic data, based on AdaGram \[[1](https://github.com/sbos/AdaGram.jl), [2](https://github.com/lopuhin/python-adagram)\].

## Requirements
### Baseline Models
- cycler          0.10.0 
- joblib          0.13.2 
- kiwisolver      1.1.0  
- matplotlib      3.1.0  
- numpy           1.16.3 
- pandas          0.24.2 
- pep8            1.7.1  
- Pillow          6.0.0  
- pip             19.1.1 
- pyparsing       2.4.0  
- python-dateutil 2.8.0  
- pytz            2019.1 
- scikit-learn    0.21.2 
- scipy           1.3.0  
- setuptools      41.0.1 
- six             1.12.0 
- torch           1.1.0  
- torchvision     0.3.0  
- tqdm            4.32.1 
- wheel           0.33.4

### DialectGram
- [Julia](https://github.com/JuliaLang/julia)
- To run DialectGram model, we need to install Python-AdaGram package from source:
  ```
  $ pip install Cython numpy
  $ pip install git+https://github.com/lopuhin/python-adagram.git
  ```
  , and install AdaGram.jl as well:
  ```
  Pkg.clone("https://github.com/sbos/AdaGram.jl.git")
  Pkg.build("AdaGram")
  ```

## Citation
Jiang, Hang*; Haoshen Hong*; Yuxing Chen*; and Vivek Kulkarni. 2019. DialectGram: Automatic Detection of Dialectal Changes with Multi-geographic Resolution Analysis. To appear in *Proceedings of the Society for Computation in Linguistics*. New Orleans: Linguistic Society of America. 

```
@inproceedings{Jiang:Hong:Chen:Kulkarni:2020:SCiL,
  Author = {Jiang, Hang  and  Hong, Haoshen  and  Chen, Yuxing  and  Kulkarni, Vivek},
  Title = {DialectGram: Automatic Detection of Dialectal Changes with Multi-geographic Resolution Analysis},
  Booktitle = {Proceedings of the Society for Computation in Linguistics},
  Location = {New Orleans},
  Publisher = {Linguistic Society of America},
  Address = {Washington, D.C.},
  Year = {2020}}
```
