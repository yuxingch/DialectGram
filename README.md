# Project Description
In this project, we use skip-gram models to capture the dialectal changes between the United States and the United Kingdom on social media.

## GEODIST-PyTorch
PyTorch version for GEODIST: the computational approach for tracking and detecting statistically significant linguistic shifts of words across geographical regions

We also include the baslien models and AdaGram model that are used in our project.

## Dataset
Twitter corpus (Geo-Tweets2019) used in this project can be found at [Geo-Twitter2019](https://github.com/emoryjianghang/Geo-Twitter2019).

## Models
- Baseline Models
  - Frequencies Model
  - Syntactic Model
- GEODIST Model: obtain region-specific word embeddings
- AdaGram Model: a wrapper of AdaGram mdoel to use in our framework

Evaluation metrics are also available.


### References

1. [Freshman or Fresher? Quantifying the Geographic Variation of Language in Online Social Media]( https://arxiv.org/pdf/1510.06786.pdf)
2. [Java code](https://github.com/dbamman/geoSGLM) for learning geographically-informed word embeddings
