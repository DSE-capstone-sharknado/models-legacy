#include "corpus.hpp"
#include "POP.hpp"
#include "WRMF.hpp"
#include "BPRMF.hpp"
#include "VBPR.hpp"
#include "TVBPR.hpp"
#include "BPRTMF.hpp"
#include "TVBPRplus.hpp"


void go_POP(corpus* corp)
{
	POP md(corp);
	double valid, test, std;
	md.AUC(&valid, &test, &std);
	fprintf(stderr, "\n\n <<< Popularity >>> Test AUC = %f, Test Std = %f\n", test, std);

	int num_item;
	md.AUC_coldItem(&test, &std, &num_item);
	fprintf(stderr, "\n\n <<< Popularity >>> Cold Start: #Item = %d, Test AUC = %f, Test Std = %f\n", num_item, test, std);
}

void go_WRMF(corpus* corp, int K, double alpha, double lambda, int iterations, const char* corp_name)
{
	WRMF md(corp, K, alpha, lambda);
	md.init();
	md.train(iterations);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_BPRMF(corpus* corp, int K, double lambda, double biasReg, int iterations, const char* corp_name)
{
	BPRMF md(corp, K, lambda, biasReg);
	md.init();
	md.train(iterations, 0.005);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_BPRTMF(corpus* corp, int K, double lambda, double biasReg, int nEpoch, int iterations, const char* corp_name)
{
	BPRTMF md(corp, K, lambda, biasReg, nEpoch);
	md.init(corp_name);
	md.train(iterations, 0.005);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_VBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int iterations, const char* corp_name)
{
	VBPR md(corp, K, K2, lambda, lambda2, biasReg);
	md.init();
	md.train(iterations, 0.005);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_TVBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int nEpoch, int iterations, const char* corp_name)
{
	TVBPR md(corp, K, K2, lambda, lambda2, biasReg, nEpoch);
	md.init();
	md.train(iterations, 0.005);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_TVBPRplus(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int nEpoch, int iterations, const char* corp_name)
{
	TVBPRplus md(corp, K, K2, lambda, lambda2, biasReg, nEpoch);
	md.init(corp_name);
	md.train(iterations, 0.005);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

int main(int argc, char** argv)
{
	srand(0);
	
	if (argc != 12) {
		printf(" Parameters as following: \n");
		printf(" 1. Review file path\n");
		printf(" 2. Img feature path\n");
		printf(" 3. Latent Feature Dim. (K)\n");
		printf(" 4. Visual Feature Dim. (K')\n");
		printf(" 5. alpha (for WRMF only)\n");
		printf(" 6. biasReg (regularizer for bias terms)\n");
		printf(" 7. lambda  (regularizer for general terms)\n");
		printf(" 8. lambda2 (regularizer for \"sparse\" terms)\n");
		printf(" 9. #Epoch (number of epochs) \n");
		printf("10. Max #iter \n");
		printf("11. Corpus/Category name under \"Clothing Shoes & Jewelry\" (e.g. Women)\n\n");
		exit(1);
	}

	char* reviewPath = argv[1];
	char* imgFeatPath = argv[2];
	int K  = atoi(argv[3]);
	int K2 = atoi(argv[4]);
	double alpha = atof(argv[5]);
	double biasReg = atof(argv[6]);
	double lambda = atof(argv[7]);
	double lambda2 = atof(argv[8]);
	double nEpoch = atoi(argv[9]);
	int iter = atoi(argv[10]);
	char* corp_name = argv[11];

	fprintf(stderr, "{\n");
	fprintf(stderr, "  \"corpus\": \"%s\",\n", reviewPath);

	corpus corp;
	corp.loadData(reviewPath, imgFeatPath, 5, 0);

  go_POP(&corp);
	// go_WRMF(&corp, K, alpha, lambda, iter, corp_name);
  go_BPRMF(&corp, K, lambda, biasReg, iter, corp_name);
	// go_BPRTMF(&corp, K, lambda, biasReg, nEpoch, iter, corp_name);
	
  go_VBPR(&corp, K, K2, lambda, lambda2, biasReg, iter, corp_name);
	// go_TVBPR(&corp, K, K2, lambda, lambda2, biasReg, nEpoch, iter, corp_name);
  // go_TVBPRplus(&corp, K, K2, lambda, lambda2, biasReg, nEpoch, iter, corp_name);

	corp.cleanUp();
	fprintf(stderr, "}\n");
	fflush(stderr);

	return 0;
}
