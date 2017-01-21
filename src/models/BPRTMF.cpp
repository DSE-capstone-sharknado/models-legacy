#include "BPRTMF.hpp"

void BPRTMF::init(string subcategoryName)
{
	loadCategories("productMeta_simple.txt.gz", subcategoryName, "root", false);

	NW = K * (nUsers + nItems) + nEpoch * (nItems + nCategory);
	W = new double[NW];

	getParametersFromVector(W, &beta_item_t, &beta_category_t, &gamma_user, &gamma_item, INIT);

	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			gamma_item[i][k] = rand() * 1.0 / RAND_MAX;
		}
		for (int ep = 0; ep < nEpoch; ep ++) {
			beta_item_t[ep][i] = 0;
		}
	}
	for (int ep = 0; ep < nEpoch; ep ++) {
		for (int i = 0; i < nCategory; i ++) {
			beta_category_t[ep][i] = 0;
		}
	}

	best_epochs = new epoch [nEpoch];
	bestW = new double [NW];
	copyBestModel();
}

void BPRTMF::cleanUp()
{
	getParametersFromVector(W, &beta_item_t, &beta_category_t, &gamma_user, &gamma_item, FREE);
	
	delete [] W;
	delete [] bestW;
	delete [] best_epochs;
	delete [] itemCategoryId;
}

void BPRTMF::getParametersFromVector(	double*   g,
										double*** beta_item_t, 
										double*** beta_category_t,
										double*** gamma_user, 
										double*** gamma_item,
										action_t  action )
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		delete [] (*beta_item_t);
		delete [] (*beta_category_t);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
		*beta_item_t = new double* [nEpoch];
		*beta_category_t = new double* [nEpoch];
	}

	int ind = 0;

	for (int ep = 0; ep < nEpoch; ep ++) {
		(*beta_item_t)[ep] = g + ind;
		ind += nItems;
	}

	for (int ep = 0; ep < nEpoch; ep ++) {
		(*beta_category_t)[ep] = g + ind;
		ind += nCategory;
	}

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (BPRTMF.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double BPRTMF::prediction(int user, int item, int ep)
{
	double pred = beta_item_t[ep][item] + inner(gamma_user[user], gamma_item[item], K);
	
	int cate_id = itemCategoryId[item];
	if (cate_id != -1) {
		pred += beta_category_t[ep][cate_id];
	}
	return pred;
}

void BPRTMF::updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate)
{
	double x_uij = beta_item_t[ep][pos_item_id] - beta_item_t[ep][neg_item_id];
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);
	int pos_cate_id = itemCategoryId[pos_item_id];
	int neg_cate_id = itemCategoryId[neg_item_id];
	if (pos_cate_id != -1) {
		x_uij += beta_category_t[ep][pos_cate_id];
	}
	if (neg_cate_id != -1) {
		x_uij += - beta_category_t[ep][neg_cate_id];
	}

	double deri = 1 / (1 + exp(x_uij));

	beta_item_t[ep][pos_item_id] += learn_rate * ( deri - biasReg * beta_item_t[ep][pos_item_id]);
	beta_item_t[ep][neg_item_id] += learn_rate * (-deri - biasReg * beta_item_t[ep][neg_item_id]);

	if (pos_cate_id != -1) {
		beta_category_t[ep][pos_cate_id] += learn_rate * ( deri - biasReg * beta_category_t[ep][pos_cate_id]);
	}
	if (neg_cate_id != -1) {
		beta_category_t[ep][neg_cate_id] += learn_rate * (-deri - biasReg * beta_category_t[ep][neg_cate_id]);
	}

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double w_uf = gamma_user[user_id][f];
		double h_if = gamma_item[pos_item_id][f];
		double h_jf = gamma_item[neg_item_id][f];

		gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
		gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
		gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 10.0 * h_jf);
	}
}

void BPRTMF::train(int iterations, double learn_rate)
{
	fprintf(stderr, "%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
    int best_iter = -1;

	// SGD begins
	for (int iter = 1; iter <= iterations; iter ++) {
		
		// perform one iter of SGD
		double l_dlStart = clock_();
		oneiteration(learn_rate);
		fprintf(stderr, "Iter: %d, took %f\n", iter, clock_() - l_dlStart);

		// optimize the epoch segmentation
		if (iter % 10 == 0) {
			l_dlStart = clock_();
			DP(1000);
			fprintf(stderr, "DP took %f\n", clock_() - l_dlStart);
		}

		if (iter % 1 == 0) {
			double valid, test, std;
			AUC(&valid, &test, &std);
			fprintf(stderr, "[Valid AUC = %f], Test AUC = %f, Test Std = %f\n", valid, test, std);
			
			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (valid < bestValidAUC && iter >= best_iter + 10) {
				fprintf(stderr, "Overfitted. Exiting... \n");
				break;
			}
		}
	}

	// copy back best parameters
	for (int w = 0; w < NW; w ++) {
		W[w] = bestW[w];
	}
	for(int ep = 0; ep < nEpoch; ep ++) {
		epochs[ep] = best_epochs[ep];
	}

	double valid, test, std;
	int num;
	AUC(&valid, &test, &std);
	fprintf(stderr, "\n\n === Valid AUC = %f, Test AUC = %f, Test Std = %f\n", valid, test, std);
	
	AUC_coldItem(&test, &std, &num);
	fprintf(stderr, "\n\n === Cold Start:     Test AUC = %f, Test Std = %f\n", test, std);
}

string BPRTMF::toString()
{
	char str[10000];
	sprintf(str, "BPR-TMF__nEpoch_%d_K_%d_lambda_%.2f_biasReg_%.2f", nEpoch, K, lambda, biasReg);
	return str;
}
