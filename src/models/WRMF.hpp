#pragma once

#include "model.hpp"
#include <armadillo>

using namespace arma;

class WRMF : public model
{
public:
	WRMF(corpus* corp, int K, double alpha, double lambda) 
		: model(corp)
		, K(K)
		, alpha(alpha)
		, lambda(lambda) {}

	~WRMF(){}
	
	void init();
	void cleanUp();

	double prediction(int user, int item);

	void Iterate();
	void ComputeSquareMatrix(mat& H);
	void Optimize(int u, map<int,long long>* pos_per, mat& W, mat& H);
	void train(int iterations);
	
	string toString();

	void copyBestModel();
	void loadModel(const char* path);

	/* auxiliary variables */
	mat X;
	mat Y;

	/* hyper-parameters */
	int K;
	double alpha;
	double lambda;

	/* helper variables */
	mat HH;
};
