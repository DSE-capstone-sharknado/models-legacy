#include "TimeBPR.hpp"


int TimeBPR::timeInEpoch(long long timestamp)
{
	long long interval = (voteTime_max - voteTime_min) / nBin;
	int bin_idx = min(nBin - 1, int((timestamp - voteTime_min) / interval));
	for (int i = 0; i < nEpoch; i ++) {
		if (epochs[i].bin_to >= bin_idx) {
			return i;
		}
	}
	return 0;
}

void TimeBPR::oneiteration(double learn_rate)
{
	// uniformally sample users in order to approximatelly optimize AUC for all users
	int user_id, pos_item_id, neg_item_id;

	// working memory
	vector<int>* user_matrix = new vector<int> [nUsers];
	for (int u = 0; u < nUsers; u ++) {
		for (map<int,long long>::iterator it = pos_per_user[u].begin(); it != pos_per_user[u].end(); it ++) {
			user_matrix[u].push_back(it->first);
		}
	}

	// now it begins!
	for (int i = 0; i < num_pos_events; i++) {
		
		// sample user
		user_id = sampleUser();
		vector<int>& user_items = user_matrix[user_id];

		// reset user if already exhausted
		if (user_items.size() == 0) {
			for (map<int,long long>::iterator it = pos_per_user[user_id].begin(); it != pos_per_user[user_id].end(); it ++) {
				user_items.push_back(it->first);
			}
		}

		// sample positive item
		int rand_num = rand() % user_items.size();
		pos_item_id = user_items.at(rand_num);
		user_items.at(rand_num) = user_items.back();
		user_items.pop_back();

		// sample negative item
		do {
			neg_item_id = rand() % nItems;
		} while (pos_per_user[user_id].find(neg_item_id) != pos_per_user[user_id].end());

		// now got tuple (user_id, pos_item, neg_item)
		updateFactors(user_id, pos_item_id, neg_item_id, timeInEpoch(pos_per_user[user_id][pos_item_id]), learn_rate);
	}

	delete [] user_matrix;
}

void TimeBPR::train(int iterations, double learn_rate)
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

		if (iter % 100 == 0) {
			l_dlStart = clock_();
			DP(1000);
			fprintf(stderr, "DP took %f\n", clock_() - l_dlStart);

			double valid, test, std;
			AUC(&valid, &test, &std);
			fprintf(stderr, "[Valid AUC = %f], Test AUC = %f, Test Std = %f\n", valid, test, std);
			
			if (bestValidAUC < valid) {
				bestValidAUC = valid;
				best_iter = iter;
				copyBestModel();
			} else if (valid < bestValidAUC && iter >= best_iter + 60) {
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
	fprintf(stderr, "\n\n === [All Item]: Valid AUC = %f, Test AUC = %f, Test Std = %f\n", valid, test, std);
	
	AUC_coldItem(&test, &std, &num);
	fprintf(stderr, "\n\n === [Cold Start]: Test AUC = %f, Test Std = %f\n", test, std);
}

string TimeBPR::toString()
{
	char str[10000];
	sprintf(str, "TimeBPR__nEpoch_%d_K_%d_lambda_%.2f_biasReg_%.2f", nEpoch, K, lambda, biasReg);
	return str;
}

void TimeBPR::AUC(double* AUC_val, double* AUC_test, double* std)
{
	double* AUC_u_val = new double[nUsers];
	double* AUC_u_test = new double[nUsers];

	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int epoch_test = timeInEpoch(test_per_user[u].second);
		int item_val  = val_per_user[u].first;
		int epoch_val = timeInEpoch(val_per_user[u].second);
		double x_u_test = prediction(u, item_test, epoch_test);
		double x_u_val  = prediction(u, item_val, epoch_val);

		int count_val = 0;
		int count_test = 0;
		int max = 0;
		for (int j = 0; j < nItems; j ++) {
			if (pos_per_user[u].find(j) != pos_per_user[u].end() ||  // in training set
				item_test == j ||  // in test set
				item_val == j) {   // in val set 
				continue;
			}
			max ++;
			double x_uj_test = prediction(u, j, epoch_test);
			double x_uj_val = x_uj_test;
			if (epoch_test != epoch_val) {
				x_uj_val = prediction(u, j, epoch_val);
			}
			if (x_u_test > x_uj_test) {
				count_test ++;
			}
			if (x_u_val > x_uj_val) {
				count_val ++;
			}			
		}
		AUC_u_val[u] = 1.0 * count_val / max;
		AUC_u_test[u] = 1.0 * count_test / max;
	}

	// sum up AUC
	*AUC_val = 0;
	*AUC_test = 0;
	for (int u = 0; u < nUsers; u ++) {
		*AUC_val += AUC_u_val[u];
		*AUC_test += AUC_u_test[u];
	}
	*AUC_val /= nUsers;
	*AUC_test /= nUsers;

	// calculate standard deviation
	double variance = 0;
	for (int u = 0; u < nUsers; u ++) {
		variance += square(AUC_u_test[u] - *AUC_test);
	}
	*std = sqrt(variance/nUsers);

	delete [] AUC_u_test;
	delete [] AUC_u_val;
}

void TimeBPR::AUC_coldItem(double* AUC_test, double* std, int* num_user)
{
	double* AUC_u_test = new double[nUsers];
	for (int u = 0; u < nUsers; u ++) {
		AUC_u_test[u] = -1;  // denote not testing
	}

	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val = val_per_user[u].first;
		int epoch_test = timeInEpoch(test_per_user[u].second);
		
		if (pos_per_item[item_test].size() > 5) {
			continue;
		}

		double x_u_test = prediction(u, item_test, epoch_test);

		int count_test = 0;
		int max = 0;
		for (int j = 0; j < nItems; j ++) {
			if (pos_per_user[u].find(j) != pos_per_user[u].end() ||  // in training set
				item_test == j ||  // in test set
				item_val == j) {   // in val set 
				continue;
			}
			max ++;
			double x_uj = prediction(u, j, epoch_test);
			if (x_u_test > x_uj) {
				count_test ++;
			}
		}
		AUC_u_test[u] = 1.0 * count_test / max;
	}

	// sum up AUC
	*AUC_test = 0;
	*num_user = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			*AUC_test += AUC_u_test[u];
			(*num_user) ++;
		}
	}
	*AUC_test /= (*num_user);

	// calculate standard deviation
	double variance = 0;
	for (int u = 0; u < nUsers; u ++) {
		if (AUC_u_test[u] != -1) {
			variance += square(AUC_u_test[u] - *AUC_test);
		}
	}
	*std = sqrt(variance / (*num_user));

	delete [] AUC_u_test;
}

void TimeBPR::DP(int neg_per_pos)
{
	map<pair<int,int>, vector<int> > sampleMap;	
	for (int b = 0; b < nBin; b ++) {
		for (vector<pair<int,int> >::iterator it = votes_per_bin[b].begin(); it != votes_per_bin[b].end(); it ++) {
			pair<int,int> ui = *it;

			int neg_item_id;
			for (int x = 0; x < neg_per_pos; x ++) {
				do {
					neg_item_id = rand() % nItems;
				//} while (pos_per_user[u].find(neg_item_id) != pos_per_user[u].end());
				} while (pos_per_user[ui.first].find(neg_item_id) != pos_per_user[ui.first].end() || neg_item_id == ui.second);
				sampleMap[ui].push_back(neg_item_id);
			}
		}
	}

	for (int i = 0; i < nBin; i ++) {
		for (int j = 0; j < nBin; j ++) {
			for (int k = 0; k < nEpoch; k ++) {
				for (int x = 0; x < nEpoch; x ++) {
					memo[i][j][k][x] = DBL_MAX; // not calculated
					sol[i][j][k][x] = -1; // not calculated
				}
			}
		}
	}

	fprintf(stderr, "\n\n  Re-shuffling ... \n");
	double fval = f(0, nBin - 1, 0, nEpoch, sampleMap);
	fprintf(stderr, "  Max f() = %f\n", fval);
	
	epoch* new_epochs = new epoch[nEpoch];
	
	int start = 0;
	int end = nBin - 1;
	int ep = 0;
	int pieces = nEpoch;
	int last_bin_to = -1;
	
	// Recover the optimal solution
	for(int i = 0; i < nEpoch; i ++) {
		int separator = sol[start][end][ep][pieces-1];

		if (separator == -1) { 
			fprintf(stderr, "\n\n  Exception: No solution found by DP.\n");
			exit(1);
		}

		new_epochs[i].bin_from = last_bin_to + 1;
		new_epochs[i].bin_to = last_bin_to = separator;
		
		start = separator + 1;
		pieces --;
		ep ++;
	}

	// Copy new epochs
	fprintf(stderr, "\n  ===== New epochs =====\n");
	for(int ep = 0; ep < nEpoch; ep ++) {
		epochs[ep] = new_epochs[ep];
		fprintf(stderr, "  %d-%d", epochs[ep].bin_from, epochs[ep].bin_to);
		if (ep < nEpoch - 1) {
			fprintf(stderr, ", ");
		}
	}
	fprintf(stderr, "\n");
	delete [] new_epochs;
}

double TimeBPR::onePieceVal(int start_bin, int end_bin, int ep, map<pair<int,int>, vector<int> >& sampleMap)
{
	int NT = omp_get_max_threads();
	double* llThread = new double [NT];

	double res = 0;
	for (int i = start_bin; i <= end_bin; i ++) {
		// already calculated
		if (memo[i][i][ep][0] != DBL_MAX) { 
			res += memo[i][i][ep][0];
			continue;
		}

		for (int t = 0; t < NT; t ++) {
			llThread[t] = 0;
		}

		#pragma omp parallel for schedule(dynamic)
		for (unsigned j = 0; j < votes_per_bin[i].size(); j ++) {
			int tid = omp_get_thread_num();
			
			pair<int,int> pos_pair = votes_per_bin[i].at(j);
			double x_ui = prediction(pos_pair.first, pos_pair.second, ep);

			for (vector<int>::iterator it = sampleMap[pos_pair].begin(); it != sampleMap[pos_pair].end(); it ++) {
				int neg_item_id = *it;
				double x_uj = prediction(pos_pair.first, neg_item_id, ep);
				llThread[tid] += log(sigmoid(x_ui - x_uj));
			}
		}
		double total = 0;
		for (int t = 0; t < NT; t ++) {
			total += llThread[t];
		}
		// record in memo
		memo[i][i][ep][0] = total;
		sol[i][i][ep][0] = i;
		res += total;
	}
	delete [] llThread;
	return res;
}

double TimeBPR::f(int start_bin, int end_bin, int ep, int pieces, map<pair<int,int>, vector<int> >& sampleMap)
{
	if (memo[start_bin][end_bin][ep][pieces-1] != DBL_MAX) { // already calculated
		return memo[start_bin][end_bin][ep][pieces-1];
	}
	if (pieces == 1) {
		memo[start_bin][end_bin][ep][0] = onePieceVal(start_bin, end_bin, ep, sampleMap);
		sol[start_bin][end_bin][ep][0] = memo[start_bin][end_bin][ep][0] < DBL_MAX ? end_bin : -1;
		return memo[start_bin][end_bin][ep][0];
	}

	double max = -DBL_MAX;
	for (int k = start_bin; k <= end_bin - pieces + 1; k ++) {
		double val = f(start_bin, k, ep, 1, sampleMap) + f(k + 1, end_bin, ep + 1, pieces - 1, sampleMap);
		if (val > max) {
			max = val;
			sol[start_bin][end_bin][ep][pieces-1] = k;
		}
	}
	memo[start_bin][end_bin][ep][pieces-1] = max;
	return max;
}

void TimeBPR::copyBestModel()
{
	model::copyBestModel();
	for(int ep = 0; ep < nEpoch; ep ++) {
		best_epochs[ep] = epochs[ep];
	}
}

void TimeBPR::saveModel(const char* path)
{
	FILE* f = fopen_(path, "w");
	fprintf(f, "{\n");
	fprintf(f, "  \"NW\": %d,\n", NW);

	fprintf(f, "  \"W\": [");
	for (int w = 0; w < NW; w ++) {
		fprintf(f, "%f", bestW[w]);
		if (w < NW - 1) fprintf(f, ", ");
	}
	fprintf(f, "]\n");

	fprintf(f, "  \"nEpoch\": %d,\n", nEpoch);

	fprintf(f, "  \"epochs\": [");
	for (int ep = 0; ep < nEpoch; ep ++) {
		fprintf(f, "%d, %d", best_epochs[ep].bin_from, best_epochs[ep].bin_to);
		if (ep < nEpoch - 1) fprintf(f, ", ");
	}
	fprintf(f, "]\n");

	fprintf(f, "}\n");
	fclose(f);

	fprintf(stderr, "\nModel saved to %s.\n", path);
}

void TimeBPR::loadModel(const char* path)
{
	fprintf(stderr, "\n  loading parameters from %s.\n", path);
	ifstream in;
	in.open(path);
	if (! in.good()){
		fprintf(stderr, "Can't read init solution from %s.\n", path);
		exit(1);
	}
	string line;
	string st;
	char ch;
	while(getline(in, line)) {
		stringstream ss(line);
		ss >> st;
		if (st == "\"NW\":") {
			int nw;
			ss >> nw;
			if (nw != NW) {
				fprintf(stderr, "NW not match.");
				exit(1);
			}
			continue;
		}

		if (st == "\"W\":") {
			ss >> ch; // skip '['
			for (int w = 0; w < NW; w ++) {
				if (! (ss >> W[w] >> ch)) {
					fprintf(stderr, "Read W[] error.");
					exit(1);
				}
			}
			continue;
		}

		if (st == "\"nEpoch\":") {
			int nep;
			ss >> nep;
			if (nep != nEpoch) {
				fprintf(stderr, "nEpoch not match.");
				exit(1);
			}
			continue;
		}

		if (st == "\"epochs\":") {
			ss >> ch; // skip '['
			for (int ep = 0; ep < nEpoch; ep ++) {
				if (! (ss >> epochs[ep].bin_from >> ch >> epochs[ep].bin_to >> ch)) {
					fprintf(stderr, "Read epochs[] error.");
					exit(1);
				}
			}
			continue;
		}
	}
	in.close();
}

void TimeBPR::printEpochs()
{
	long int interval = (voteTime_max - voteTime_min) / nBin;
	for(int ep = 0; ep < nEpoch; ep ++) {
		long int epoch_start = voteTime_min + interval * epochs[ep].bin_from;
		long int epoch_end = voteTime_min + interval * (epochs[ep].bin_to + 1);

		// convert to readable format
		char date_start[20];
		char date_end[20];
		struct tm *tm = gmtime(&epoch_start);
		strftime(date_start, sizeof(date_start), "%Y-%m-%d", tm);
		tm = gmtime(&epoch_end);
		strftime(date_end, sizeof(date_end), "%Y-%m-%d", tm);
		fprintf(stderr, "Epoch %d: %s(UTC) -- %s(UTC)\n", ep, date_start, date_end);
	}
}