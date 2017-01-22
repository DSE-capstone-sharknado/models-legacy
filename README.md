# Amazon Models Reference Impls.

Includes reference implementations for BMR, VBMR and TVBMP (ups and downs).

You can't build this out-of-the-box on a mac unless you have a traditional gcc tool-chain setup. I'm building it w/ a Docker image i've included in this repo. 


## Preprocessing

All the models need the ratings dataset in a simplified format, use the `convert_to_simple.py` util to do this:

```
python convert_to_simple.py reviews_Clothing_Shoes_and_Jewelry.json.gz reviews_simple.gz
```

## Build Suite

To build the suite, run the `Makefile`, by typing `make` in the project directory. This will build a binary named `train`.


## BMR Model

This is the simple model w/ no image or temporal features. It uses rating data to imply positive feedback (a purchase). Uses the BMR cost function that is minimized by SGD.

pairwise Bayesian Personalized Ranking (BPR) loss (Rendle et al., 2009) for ranking. 

### Training

To train the BMR model, pass in the following args:

* processed reviews file path
* K latent factors
* regulation coreff 1
* regulation coreff 2
* max SGD iterations constant

```
./train simple_out.gz simple_out.gz 20 k2 alpha 10 10 lambda2 epoch 10 "Clothing"
```
