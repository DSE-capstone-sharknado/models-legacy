#include "TVBPRplus.hpp"


void TVBPRplus::init(const char* subcategoryName)
{
	loadCategories("productMeta_simple.txt.gz", subcategoryName, "root", false);

	NW = nEpoch * (nCategory + nItems) 
		 + K * (nItems + nUsers) // latent factors
		 + K2 * nUsers           // visual factors
		 + (K2 + 1) * corp->imFeatureDim // static visual parameters
		 + nEpoch * (K2 + corp->imFeatureDim) // dynamic weights
		 + nEpoch * (K2 + 1) * corp->imFeatureDim;

	W = new double [NW];

	getParametersFromVector(W, &beta_category_t, &beta_item_t, &gamma_user, &gamma_item, &theta_user, &U, &beta_cnn, &J_t, &C_t, &U_t, &beta_cnn_t, INIT);

	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			gamma_user[u][k] = 1.0 * rand() / RAND_MAX;
		}
		for (int k = 0; k < K2; k ++) {
			theta_user[u][k] = 1.0 * rand() / RAND_MAX;
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			gamma_item[i][k] = 1.0 * rand() / RAND_MAX;
		}
		for (int ep = 0; ep < nEpoch; ep ++) {
			beta_item_t[ep][i] = 0;
		}
	}

	// initialize all matrixes to be same
	for (int r = 0; r < K2; r ++) {
		for (int c = 0; c < corp->imFeatureDim; c ++) {
			double rand_num = 1.0 * rand() / RAND_MAX;
			for (int ep = 0; ep < nEpoch; ep ++) {
				U_t[ep][r][c] = rand_num;
			}
			U[r][c] = 1.0 * rand() / RAND_MAX;
		}
	}
	for (int i = 0; i < corp->imFeatureDim; i ++) {
		beta_cnn[i] = 1.0 * rand() / RAND_MAX;
	}
	for (int ep = 0; ep < nEpoch; ep ++) {
		for (int i = 0; i < nCategory; i ++) {
			beta_category_t[ep][i] = 0;
		}
		for (int i = 0; i < corp->imFeatureDim; i ++) {
			beta_cnn_t[ep][i] = 0;
		}
		for (int k = 0; k < K2; k ++) {
			J_t[ep][k] = 0;
		}
		for (int c = 0; c < corp->imFeatureDim; c ++) {
			C_t[ep][c] = 0;
		}
	}
	
	best_epochs = new epoch [nEpoch];
	bestW = new double [NW];
	copyBestModel();

	/* for speed up */
	theta_item_per_ep = new double** [nEpoch];
	for (int ep = 0; ep < nEpoch; ep ++) {
		theta_item_per_ep[ep] = new double* [nItems];
		for (int i = 0; i < nItems; i ++) {
			theta_item_per_ep[ep][i] = new double [K2];
		}
	}
	beta_item_visual_per_ep = new double* [nEpoch];
	for (int ep = 0; ep < nEpoch; ep ++) {
		beta_item_visual_per_ep[ep] = new double [nItems];
	}
}

void TVBPRplus::cleanUp()
{
	getParametersFromVector(W, &beta_category_t, &beta_item_t, &gamma_user, &gamma_item, &theta_user, &U, &beta_cnn, &J_t, &C_t, &U_t, &beta_cnn_t, FREE);

	delete [] W;
	delete [] bestW;
	delete [] best_epochs;

	for (int ep = 0; ep < nEpoch; ep ++) {
		for (int i = 0; i < nItems; i ++) {
			delete [] theta_item_per_ep[ep][i];
		}
		delete theta_item_per_ep[ep];
	}
	delete [] theta_item_per_ep;

	for (int ep = 0; ep < nEpoch; ep ++) {
		delete [] beta_item_visual_per_ep[ep];
	}
	delete [] beta_item_visual_per_ep;	
	delete [] itemCategoryId;
}

void TVBPRplus::getParametersFromVector(	double*    g,
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
											action_t   action)
{
	if (action == FREE) {
		delete [] (*beta_category_t);
		delete [] (*beta_item_t);
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		delete [] (*theta_user);
		delete [] (*U);
		delete [] (*J_t);
		delete [] (*C_t);
		for (int i = 0; i < nEpoch; i ++) {
			delete [] (*U_t)[i];
		}
		delete [] (*U_t);
		delete [] (*beta_cnn_t);
		return;
	}

	if (action == INIT)	{
		*beta_category_t = new double* [nEpoch];
		*beta_item_t = new double* [nEpoch];
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
		*theta_user = new double* [nUsers];
		*U = new double* [K2];
		*J_t = new double* [nEpoch];
		*C_t = new double* [nEpoch];
		*U_t = new double** [nEpoch];
		for (int i = 0; i < nEpoch; i ++) {
			(*U_t)[i] = new double* [K2];
		}
		*beta_cnn_t = new double* [nEpoch];
	}

	int ind = 0;

	for (int i = 0; i < nEpoch; i ++) {
		(*beta_category_t)[i] = g + ind;
		ind += nCategory;
	}

	for (int i = 0; i < nEpoch; i ++) {
		(*beta_item_t)[i] = g + ind;
		ind += nItems;
	}

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}

	for (int u = 0; u < nUsers; u ++) {
		(*theta_user)[u] = g + ind;
		ind += K2;
	}

	for (int k = 0; k < K2; k ++) {
		(*U)[k] = g + ind;
		ind += corp->imFeatureDim;
	}

	*beta_cnn = g + ind;
	ind += corp->imFeatureDim;

	for (int i = 0; i < nEpoch; i ++) { 
		(*J_t)[i] = g + ind;
		ind += K2;
	}

	for (int i = 0; i < nEpoch; i ++) {
		(*C_t)[i] = g + ind;
		ind += corp->imFeatureDim;
	}

	for (int i = 0; i < nEpoch; i ++) { 
		for (int k = 0; k < K2; k ++) {
			(*U_t)[i][k] = g + ind;
			ind += corp->imFeatureDim;
		}
	}

	for (int i = 0; i < nEpoch; i ++) {
		(*beta_cnn_t)[i] = g + ind;
		ind += corp->imFeatureDim;
	}

	if (ind != NW) {
		printf("Got bad index (TVBPRplus.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double TVBPRplus::prediction(int user, int item, int ep)
{
	double pred = beta_item_t[ep][item] + 
				+ inner(gamma_user[user], gamma_item[item], K)
				+ inner(theta_user[user], theta_item_per_ep[ep][item], K2)
				+ beta_item_visual_per_ep[ep][item];

	int cate_id = itemCategoryId[item];
	if (cate_id != -1) {
		pred += beta_category_t[ep][cate_id];
	}
	return pred;
}

void TVBPRplus::updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate)
{
	// sparse representation of f_i - f_j
	vector<pair<int, float> > diff;
	vector<pair<int, float> >& feat_i = corp->imageFeatures[pos_item_id];
	vector<pair<int, float> >& feat_j = corp->imageFeatures[neg_item_id];
	unsigned p_i = 0, p_j = 0;
	while (p_i < feat_i.size() && p_j < feat_j.size()) {
		int ind_i = feat_i[p_i].first;
		int ind_j = feat_j[p_j].first;
		if (ind_i < ind_j) {
			diff.push_back(make_pair(ind_i, feat_i[p_i].second));
			p_i ++;
		} else if (ind_i > ind_j) {
			diff.push_back(make_pair(ind_j, - feat_j[p_j].second));
			p_j ++;
		} else {
			diff.push_back(make_pair(ind_i, feat_i[p_i].second - feat_j[p_j].second));
			p_i ++; p_j ++;
		}
	}
	while (p_i < feat_i.size()) {
		diff.push_back(feat_i[p_i]);
		p_i ++;
	}
	while (p_j < feat_j.size()) {
		diff.push_back(make_pair(feat_j[p_j].first, - feat_j[p_j].second));
		p_j ++;
	}

	// U_{t} * (f_i - f_j)
	for (int r = 0; r < K2; ++ r) {
		theta_item_per_ep[0][0][r] = 0;  // borrow the memory at index 0
		theta_item_per_ep[0][1][r] = 0;  // borrow the memory at index 1	
		for (unsigned ind = 0; ind < diff.size(); ind ++) {
			int c = diff[ind].first;
			theta_item_per_ep[0][0][r] += U[r][c] * diff[ind].second;
			theta_item_per_ep[0][1][r] += U_t[ep][r][c] * diff[ind].second;
		}
	}
	double visual_score = 0;
	for (int r = 0; r < K2; ++ r) {
		visual_score += theta_user[user_id][r] * (theta_item_per_ep[0][0][r] * J_t[ep][r] + theta_item_per_ep[0][1][r]);
	}
	double visual_bias = 0;
	for (unsigned ind = 0; ind < diff.size(); ind ++) {
		int c = diff[ind].first;
		visual_bias += (beta_cnn[c] * C_t[ep][c] + beta_cnn_t[ep][c]) * diff[ind].second;
	}

	// x_uij = prediction(user_id, pos_item_id, ep) - prediction(user_id, neg_item_id, ep);
	double x_uij = beta_item_t[ep][pos_item_id] - beta_item_t[ep][neg_item_id];
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);
	x_uij += visual_score + visual_bias;

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
		beta_category_t[ep][pos_cate_id] += learn_rate * (deri - biasReg * beta_category_t[ep][pos_cate_id]);
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

	// adjust visual factors
	for (int r = 0; r < K2; r ++) {
		double v_uf = theta_user[user_id][r];
		double j_tf = J_t[ep][r];

		for (unsigned ind = 0; ind < diff.size(); ind ++) {
			int c = diff[ind].first;
			double common_part = deri * v_uf * diff[ind].second;
			U[r][c] += learn_rate * (common_part * j_tf);
			U_t[ep][r][c] += learn_rate * (common_part - lambda2 * U_t[ep][r][c]);
		}
		theta_user[user_id][r] += learn_rate * (deri * (theta_item_per_ep[0][0][r] * j_tf + theta_item_per_ep[0][1][r]) - lambda  * v_uf);
		J_t[ep][r] += learn_rate * (deri * theta_item_per_ep[0][0][r] * v_uf - lambda2 * j_tf);
	}
	for (unsigned ind = 0; ind < diff.size(); ind ++) {
		int c = diff[ind].first;
		double c_tf  = C_t[ep][c];
		double b_cnn = beta_cnn[c];
		double common_part = learn_rate * deri * diff[ind].second;
		beta_cnn[c] += common_part * c_tf;
		C_t[ep][c]  += common_part * b_cnn - learn_rate * lambda2 * c_tf;
		beta_cnn_t[ep][c] += common_part - learn_rate * lambda2 * beta_cnn_t[ep][c];
	}
}

string TVBPRplus::toString()
{
	char str[10000];
	sprintf(str, "TVBPRplus__nEpoch_%d_K_%d_K2_%d_lambda_%.2f_lambda2_%.6f_biasReg_%.2f", nEpoch, K, K2, lambda, lambda2, biasReg);
	return str;
}

void TVBPRplus::saveModel(const char* path)
{
	FILE* f = fopen_(path, "w");
	fprintf(f, "{\n");
	fprintf(f, "  \"NW\": %d,\n", NW);

	fprintf(f, "  \"beta_category_t(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int k = 0; k < nCategory; k ++) {
			fprintf(f, "%f,", beta_category_t[ep][k]);
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"beta_item_t(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int k = 0; k < nItems; k ++) {
			fprintf(f, "%f,", beta_item_t[ep][k]);
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"gamma_user\": [");
	for (int i = 0; i < nUsers; i ++) {
		for (int k = 0; k < K; k ++) {
			fprintf(f, "%f,", gamma_user[i][k]);
		}
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"gamma_item\": [");
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			fprintf(f, "%f,", gamma_item[i][k]);
		}
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"theta_user\": [");
	for (int i = 0; i < nUsers; i ++) {
		for (int k = 0; k < K2; k ++) {
			fprintf(f, "%f,", theta_user[i][k]);
		}
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"U\": [");
	for (int r = 0; r < K2; r ++) {
		for (int c = 0; c < 4096; c ++) {
			fprintf(f, "%f,", U[r][c]);
		}
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"U(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int r = 0; r < K2; r ++) {
			for (int c = 0; c < 4096; c ++) {
				fprintf(f, "%f,", U_t[ep][r][c]);
			}
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"beta_cnn\": [");
	for (int c = 0; c < 4096; c ++) {
		fprintf(f, "%f,", beta_cnn[c]);
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"beta_cnn(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int c = 0; c < 4096; c ++) {
			fprintf(f, "%f,", beta_cnn_t[ep][c]);
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"J(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int k = 0; k < K2; k ++) {
			fprintf(f, "%f,", J_t[ep][k]);
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"C(t)\": [\n");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "  \"epoch %d\": [", ep);
		for (int c = 0; c < 4096; c ++) {
			fprintf(f, "%f,", C_t[ep][c]);
		}
		fprintf(f, "]\n");
	}
	fprintf(f, "]\n");

	fprintf(f, "}\n");
	fclose(f);

	fprintf(stderr, "\nMy Model saved to %s.\n", path);
}