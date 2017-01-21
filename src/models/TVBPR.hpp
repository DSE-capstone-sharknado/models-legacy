#pragma once

#include "TimeBPR.hpp"

class TVBPR : public TimeBPR
{
public:
	TVBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int nEpoch) 
		: TimeBPR(corp, K, lambda, biasReg, nEpoch)
		, K2(K2)
		, lambda2(lambda2) {}
	
	~TVBPR(){}
	
	void init();
	void cleanUp();

	void getVisualFactors();
	double prediction(int user, int item, int ep);

	void getParametersFromVector(	double*    g,
									double**   beta_item,
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

	// Dynamic Programming
	void DP(int neg_per_pos);

	void AUC(double* AUC_val, double* AUC_test, double* std);
	void AUC_coldItem(double* AUC_test, double* std, int* num_user);

	void updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate);

	string toString();

	/* hyper-parameters */
	int K2;
	double lambda2;

	/* Model parameters */
	double**  theta_user;
	double**  U;
	double*   beta_cnn;
	double**  J_t;
	double**  C_t;
	double*** U_t;
	double**  beta_cnn_t;

	/* auxiliary variables for speep-up */
	double*** theta_item_per_ep;
	double**  beta_item_visual_per_ep;
};
