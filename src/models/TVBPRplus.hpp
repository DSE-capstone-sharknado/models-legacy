#pragma once

#include "TVBPR.hpp"


class TVBPRplus : public TVBPR
{
public:
	TVBPRplus(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int nEpoch) 
			: TVBPR(corp, K, K2, lambda, lambda2, biasReg, nEpoch) {}
	
	~TVBPRplus(){}
	
	void init(const char* subcategoryName);
	void cleanUp();

	double prediction(int user, int item, int ep);

	void getParametersFromVector(	double*    g,
									double***  beta_category_t,
									double***  beta_item_t,
									double***  gamma_user, 
									double***  gamma_item,
									double***  theta_user,
									double***  U,
									double**   beta_cnn,
									double***  J_t,
									double***  C_t,
									double**** U_t,
									double***  beta_cnn_t,
									action_t   action);

	void updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate);
	void saveModel(const char* path);
	string toString();

	/* Model parameters */
	double** beta_item_t;
	double** beta_category_t;
};
