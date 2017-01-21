#include "WRMF.hpp"


void WRMF::init()
{
	X.randu(nUsers, K);  // uniformly distributed number between 0 to 1
	Y.randu(nItems, K);  // uniformly distributed number between 0 to 1
	HH = mat(K, K);

	NW = (nUsers + nItems) * K; 
	bestW = new double[NW];
}

void WRMF::cleanUp()
{
	delete[] bestW;
}

double WRMF::prediction(int user, int item)
{
	return dot(X.row(user), Y.row(item));
}

void WRMF::Iterate()
{
	// perform alternating parameter fitting
	ComputeSquareMatrix(Y);
	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		Optimize(u, pos_per_user, X, Y);
	}

	ComputeSquareMatrix(X);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < nItems; i ++) {
		Optimize(i, pos_per_item, Y, X);
	}
}

void WRMF::ComputeSquareMatrix(mat& H)
{
	// mm is symmetric
	for (int f_1 = 0; f_1 < (int)H.n_cols; f_1 ++) {
		for (int f_2 = f_1; f_2 < (int)H.n_cols; f_2 ++)	{
			double d = 0;
			for (int i = 0; i < (int)H.n_rows; i++) {
				d += H(i, f_1) * H(i, f_2);
			}
			HH(f_1, f_2) = d;
			HH(f_2, f_1) = d;
		}
	}
	return;
}

void WRMF::Optimize(int u, map<int,long long>* pos_per, mat& W, mat& H)
{
	// HC_minus_IH is symmetric
	// create HC_minus_IH in O(f^2|S_u|)
	mat HC_minus_IH = mat(K, K);

	for (int f_1 = 0; f_1 < K; f_1++) {
		for (int f_2 = f_1; f_2 < K; f_2++) {
			double d = 0;
			for(map<int,long long>::iterator it = pos_per[u].begin(); it != pos_per[u].end(); it ++) {
				d += H(it->first, f_1) * H(it->first, f_2);
			}
			HC_minus_IH(f_1, f_2) = d * alpha;
			HC_minus_IH(f_2, f_1) = d * alpha;
		}
	}
	// create HCp in O(f|S_u|)
	double* HCp = new double [K];
	for (int f = 0; f < K; f++) {
		double d = 0;
		for(map<int,long long>::iterator it = pos_per[u].begin(); it != pos_per[u].end(); it ++) {
			d += H(it->first, f);
		}
		HCp[f] = d * (1 + alpha);
	}
	// create m = HH + HC_minus_IH + reg*I
	// m is symmetric
	// the inverse m_inv is symmetric
	mat m = mat(K, K);
	for (int f_1 = 0; f_1 < K; f_1 ++) {
		for (int f_2 = f_1; f_2 < K; f_2++) {
			double d = HH(f_1, f_2) + HC_minus_IH(f_1, f_2);
			if (f_1 == f_2) { 
				d += lambda;
			}
			m(f_1, f_2) = d;
			m(f_2, f_1) = d;
		}
	}

	mat m_inv = inv(m);
	// write back optimal W
	for (int f = 0; f < K; f ++) {
		double d = 0;
		for (int f_2 = 0; f_2 < K; f_2 ++) {
			d += m_inv(f, f_2) * HCp[f_2];
		}
		
		W(u, f) = d;
	}

	delete [] HCp;
}

void WRMF::train(int iterations)
{
	fprintf(stderr, "%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
	int best_iter = 0;

	for (int iter = 1; iter <= iterations; iter ++) {
		// perform one iteration of ALS
		double l_dlStart = clock_();
		Iterate();
		fprintf(stderr, "Iter: %d, took %f\n", iter, clock_() - l_dlStart);

		if (iter % 10 == 0) {
			double valid, test, std;
			AUC(&valid, &test, &std);
			fprintf(stderr, "[Valid AUC = %f], Test AUC = %f, Test Std = %f\n", valid, test, std);
			
			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (valid < bestValidAUC && iter > best_iter + 20) {
				fprintf(stderr, "Overfitted. Exiting... \n");
				break;
			}
		}
	}

	// copy back best parameters
	int w = 0;
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			X(u, k) = bestW[w ++];
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			Y(i, k) = bestW[w ++];
		}
	}

	double valid, test, std;
	int num;
	AUC(&valid, &test, &std);
	fprintf(stderr, "\n\n === [All Item]: Valid AUC = %f, Test AUC = %f, Test Std = %f\n", valid, test, std);
	
	AUC_coldItem(&test, &std, &num);
	fprintf(stderr, "\n\n === [Cold Start]: Test AUC = %f, Test Std = %f\n", test, std);
}

void WRMF::copyBestModel()
{
	int w = 0;
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			bestW[w ++] = X(u, k);
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			bestW[w ++] = Y(i, k);
		}
	}
}

/// model must be first initialized before calling this function
void WRMF::loadModel(const char* path)
{
	W = new double [NW];
	model::loadModel(path);
	// copy back best parameters
	int w = 0;
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			X(u, k) = W[w ++];
		}
	}
	for (int i = 0; i < nItems; i ++) {
		for (int k = 0; k < K; k ++) {
			Y(i, k) = W[w ++];
		}
	}
	delete [] W;
}

string WRMF::toString()
{
	char str[1000];
	sprintf(str, "WRMF__K_%d_alpha_%.2f_lambda_%.2f", K, alpha, lambda);
	return str;
}