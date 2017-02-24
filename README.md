# Amazon Models Reference Impls.

Includes reference implementations for BMR, VBMR and TVBMP (ups and downs).

You can't build this out-of-the-box on a mac unless you have a traditional gcc tool-chain setup. I'm building it w/ a Docker image i've included in this repo. * See Docker note below


## Preprocessing

All the models need the ratings dataset in a simplified format, use the `convert_to_simple.py` util to do this:

```
python convert_to_simple.py reviews_Clothing_Shoes_and_Jewelry.json.gz reviews_simple.gz
```


Then there is a script that splits the file into categories. For input it takes the gzipped simplifed output from above and a meta data file which has some category mappings: https://drive.google.com/file/d/0B9Ck8jw-TZUELVRLMVdJNTRUU0U/view?usp=sharing

```
 ./getClothingSubReviews ../simple_out.gz productMeta_simple.txt
```
 
 This will generate these segmented review files (which you need to gzip to use in the training program):
 
```
reviews_Baby.txt
reviews_Boys.txt
reviews_Girls.txt
reviews_Men.txt
reviews_Women.txt
```

However, when I train on this data it trains it trips the sanity check on line 51 of model.hpp

```
//in this case this train binary is build to train a BPR model so the fields that are n/a for BPR are passed in as na args
./train reviews_Women.txt.gz na 20 na na 10 10 na na 10 "WomenClothing"
```

```
{
  "corpus": "tools/reviews_Women.txt.gz",
  Loading votes from tools/reviews_Women.txt.gz, userMin = 5, itemMin = 0  ..

  Generating votes data
  "nUsers": 34086, "nItems": 14016, "nVotes": 171279


 Corpus split exception.
```

It takes the #users down from 39,387 to 34,086 and that seems to result in each user not having at least 3 reviews which is needed for training

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

Example Output:

```
{
  "corpus": "simple_out.gz",
  Loading votes from simple_out.gz, userMin = 5, itemMin = 0  ....

  Generating votes data
  "nUsers": 39387, "nItems": 23033, "nVotes": 278677

<<< BPR-MF__K_20_lambda_10.00_biasReg_10.00 >>>

Iter: 1, took 0.266870
Iter: 2, took 0.247799
Iter: 3, took 0.250260
Iter: 4, took 0.248863
Iter: 5, took 0.262415
[Valid AUC = 0.358993], Test AUC = 0.360745, Test Std = 0.305036
Iter: 6, took 0.245542
Iter: 7, took 0.245433
Iter: 8, took 0.236978
Iter: 9, took 0.236842
Iter: 10, took 0.234738
[Valid AUC = 0.610926], Test AUC = 0.611459, Test Std = 0.294283


 <<< BPR-MF >>> Test AUC = 0.611459, Test Std = 0.294283


 <<< BPR-MF >>> Cold Start: #Item = 11453, Test AUC = 0.554613, Test Std = 0.300705

Model saved to Clothing__BPR-MF__K_20_lambda_10.00_biasReg_10.00.txt.
}
```

##Docker

To run the docker image:

```
docker run -v ~/Development/DSE/capstone/UpsDowns:/mnt/mac  -ti updowns /bin/bash
```

the `v` flag will mount the source repot to `/mnt/mac` in the container.
