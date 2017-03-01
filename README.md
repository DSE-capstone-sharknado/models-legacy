# Amazon Models Reference Impls.

Includes reference implementations for BMR, VBMR and TVBMP (ups and downs).

You can't build this out-of-the-box on a mac unless you have a traditional gcc tool-chain setup. I'm building it w/ a Docker image i've included in this repo. * See Docker note below


## Preprocessing

All the models need the ratings dataset in a simplified format, use the `convert_to_simple.py` util to do this:

```
python convert_to_simple.py reviews_Clothing_Shoes_and_Jewelry.json.gz reviews_simple.gz
```


## Segmentation

It is useful for evaluation to segment the clothing dataset into: women, men, boys, girls and baby using the `getClothingSubReviews` script.

For input it takes the gzipped simplified output from above and a meta data file which has some category mappings: https://drive.google.com/file/d/0B9Ck8jw-TZUELVRLMVdJNTRUU0U/view?usp=sharing

```
 ./getClothingSubReviews ../clothing_5core.gz productMeta_simple.txt
 ./getClothingSubReviews ../data/clothing_full.gz ../data/productMeta_simple.txt
```
 
 This will generate these segmented review files (which you need to gzip to use in the training program):
 
```
reviews_Baby.txt
reviews_Boys.txt
reviews_Girls.txt
reviews_Men.txt
reviews_Women.txt
```

I can't seem to get this to work on 5-core as I get a sanity check error when I train.

## Build Suite

To build the suite, run the `Makefile`, by typing `make` in the project directory. This will build a binary named `train`.


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


## Models

### BPR

On Amazon, regularization hyper-parameter lambda=10 works the best for BPR-MF, MM-MF and VBPR in most cases. 