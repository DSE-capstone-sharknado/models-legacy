#pragma once

#include "BPRMF.hpp"

class TimeBPR : public BPRMF
{
public:
	TimeBPR(corpus* corp, int K, double lambda, double biasReg, int nEpoch)
		: BPRMF(corp, K, lambda, biasReg)
		, nEpoch(nEpoch)
	{
		nBin = 80;

		// calculate MIN & MAX
		voteTime_min = LLONG_MAX;
		voteTime_max = LLONG_MIN;
		for (int x = 0; x < nVotes; x++) {
			vote* vi = corp->V.at(x);
			long long voteTime = vi->voteTime;
			if (voteTime < voteTime_min) {
				voteTime_min = voteTime;
			}
			if (voteTime > voteTime_max) {
				voteTime_max = voteTime;
			}
		}

		// calculate pos_item_per_bin
		votes_per_bin = new vector<pair<int,int> > [nBin];
		long long interval = (voteTime_max - voteTime_min) / nBin;
		for (int u = 0; u < nUsers; u ++) {
			if (val_per_user[u].first != -1) {
				int item = val_per_user[u].first;
				long long voteTime = val_per_user[u].second;
				int bin_idx = min(nBin - 1, int((voteTime - voteTime_min) / interval));
				votes_per_bin[bin_idx].push_back(make_pair(u, item));
			}
		}

		// initialize epochs
		epochs = new epoch [nEpoch];
		interval = nBin / nEpoch;
		int bin_to = nBin - 1;
		for (int ep = nEpoch - 1; ep >= 0; ep --) {
			epochs[ep].bin_to = bin_to;
			if (ep == 0) {
				epochs[ep].bin_from = 0;
				break;
			} else {
				epochs[ep].bin_from = epochs[ep].bin_to - interval + 1;
				bin_to = epochs[ep].bin_from - 1;
			}
		}

		fprintf(stderr, "\n===== Initial Epoch segmentation =====\n");
		for (int ep = 0; ep < nEpoch; ep ++) {
			fprintf(stderr, "Epoch %d: %d -- %d\n", ep, epochs[ep].bin_from, epochs[ep].bin_to);
		}
		fprintf(stderr, "=======================================\n\n");

		// init DP
		memo = new double*** [nBin];
		sol = new int*** [nBin];
		for (int i = 0; i < nBin; i ++) {
			memo[i] = new double** [nBin];
			sol[i] = new int** [nBin];
			for (int j = 0; j < nBin; j ++) {
				memo[i][j] = new double* [nEpoch];
				sol[i][j] = new int* [nEpoch];
				for (int k = 0; k < nEpoch; k ++) {
					memo[i][j][k] = new double[nEpoch];
					sol[i][j][k] = new int[nEpoch];
				}
			}
		}
	}

	~TimeBPR()
	{
		delete [] votes_per_bin;
		delete [] epochs;

		for (int i = 0; i < nBin; i ++) {		
			for (int j = 0; j < nBin; j ++) {			
				for (int k = 0; k < nEpoch; k ++) {
					delete [] memo[i][j][k];
					delete [] sol[i][j][k];
				}
				delete [] memo[i][j];
				delete [] sol[i][j];
			}
			delete [] memo[i];
			delete [] sol[i];
		}
		delete [] memo;
		delete [] sol;
	}

	int timeInEpoch(long long timestamp);
	virtual double prediction(int user, int item, int ep) = 0;	
	virtual void updateFactors(int user_id, int pos_item_id, int neg_item_id, int ep, double learn_rate) = 0;
	virtual void oneiteration(double learn_rate);
	void train(int iterations, double learn_rate);

	// Evaluation
	virtual void AUC(double* AUC_val, double* AUC_test, double* std);
	virtual void AUC_coldItem(double* AUC_test, double* std, int* num_user);
	
	// Model related
	void copyBestModel();
	void loadModel(const char* path);
	void saveModel(const char* path);

	// Dynamic Programming optimizes the epoch segmentation
	virtual void DP(int neg_per_pos);
	double f(int start_bin, int end_bin, int ep, int pieces, map<pair<int,int>, vector<int> >& sampleMap);
	double onePieceVal(int start_bin, int end_bin, int ep, map<pair<int,int>, vector<int> >& sampleMap);

	// Print
	void printEpochs();

	string toString();

	long long voteTime_min;
	long long voteTime_max;

	epoch* epochs;
	epoch* best_epochs;
	int nBin;
	int nEpoch;

	// For DP
	double**** memo;
	int**** sol;
	vector<pair<int,int> >* votes_per_bin;
};
