# gpbayeskit

This toolbox is inspired by the R package GPBayes, but with extended features. 
Gaussian processes ('GPs') have been widely used to model spatial 
data, 'spatio'-temporal data, and computer experiments in diverse 
areas of statistics including spatial statistics, 'spatio'-temporal 
statistics, uncertainty quantification, and machine learning. 
This package creates basic tools for fitting and prediction based 
on 'GPs' with spatial data, 'spatio'-temporal data, and computer 
experiments. Key characteristics for this GP tool include: 
the comprehensive implementation of various covariance functions 
including the 'Matérn' family and the Confluent 'Hypergeometric' 
family with isotropic form, tensor form, and automatic relevance 
determination form, where the isotropic form is widely used in 
spatial statistics, the tensor form is widely used in design 
and analysis of computer experiments and uncertainty quantification, 
and the automatic relevance determination form is widely used in 
machine learning. 


## Install locally

```bash
pip install -e .
