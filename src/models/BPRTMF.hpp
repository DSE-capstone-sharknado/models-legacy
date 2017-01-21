#pragma once

#include "TimeBPR.hpp"
#include "categoryTree.hpp"


class BPRTMF : public TimeBPR
{
public:
	BPRTMF(corpus* corp, int K, double lambda, double biasReg, int nEpoch) 
		: TimeBPR(corp, K, lambda, biasReg, nEpoch) {}

	~BPRTMF() {}
	
	void init(string subcategoryName);
	void cleanUp();

	double prediction(int user, int item, int ep);
	void getParametersFromVector(	double*   g,
									double*** beta_item_t,
									double*** beta_category_t,
									double*** gamma_user, 
									double*** gamma_item,
									action_t  action);

	void updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate);
	void train(int iterations, double learn_rate);
	
	// Who I am
	string toString();

	/* auxiliary variables */
	double** beta_item_t;
	double** beta_category_t;
};
