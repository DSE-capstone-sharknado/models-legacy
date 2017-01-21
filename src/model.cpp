#include "model.hpp"

// Parse category info for all products
void model::loadCategories(const char* categoryPath, string subcategoryName, string rootName, bool skipRoot)
{
	itemCategoryId = new int [nItems];
	for (int i = 0; i < nItems; i ++) {
		itemCategoryId[i] = -1;
	}

	fprintf(stderr, "\n  Loading category data");
	categoryTree* ct = new categoryTree(rootName, skipRoot);

	igzstream in;
	in.open(categoryPath);
	if (! in.good()) {
		fprintf(stderr, "\n  Can't load category from %s.\n", categoryPath);
		exit(1);
	}

	string line;

	int item = -1;
	int count = 0;
	nCategory = 0;

	while (getline(in, line)) {
		istringstream ss(line);

		if (line.c_str()[0] != ' ') {
			string itemId;
			double price = -1;
			string brand("unknown_brand");
			ss >> itemId >> price >> brand;
			if (corp->itemIds.find(itemId) == corp->itemIds.end()) {
				item = -1;
				continue;
			}

			item = corp->itemIds[itemId];
			itemPrice[item] = price;
			itemBrand[item] = brand;

			// print process
			count ++;
			if (not (count % 10000)) {
				fprintf(stderr, ".");
			}
			continue;
		}

		if (item == -1) {
			continue;
		}
		vector<string> category;

		// Category for each product is a comma-separated list of strings
		string cat;
		while (getline(ss, cat, ',')) {
			category.push_back(trim(cat));
		}

		if (category[0] != "Clothing Shoes & Jewelry" || category[1] != subcategoryName) {
			continue;
		}

		ct->addPath(category);

		if (category.size() < 4) {
			continue;
		}
		string* categoryP = &(category[0]);
		categoryNode* targetNode = ct->root->find(categoryP, 4);
		if (targetNode == 0) {
			fprintf(stderr, "Can't find the category node.\n");
			exit(1);
		}

		if (nodeIds.find(targetNode->nodeId) == nodeIds.end()) {
			nodeIds[targetNode->nodeId] = nCategory;
			rNodeIds[nCategory] = category[2] + "," + category[3];
			nCategory ++;
		}
		itemCategoryId[item] = nodeIds[targetNode->nodeId];
	}

	fprintf(stderr, "\n");
	in.close();

	int total = 0;
	for (int i = 0; i < nItems; i ++) {
		if (itemCategoryId[i] != -1) {
			total ++;
		}
	}
	fprintf(stderr, "  #Items with category: %d\n", total);
	if (1.0 * total / nItems < 0.5) {
		fprintf(stderr, "So few items are having category info. Sth wrong may have happened.\n");
		exit(1);
	}
}

void model::AUC(double* AUC_val, double* AUC_test, double* std)
{
	double* AUC_u_val = new double[nUsers];
	double* AUC_u_test = new double[nUsers];

	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;;
		double x_u_test = prediction(u, item_test);
		double x_u_val  = prediction(u, item_val);

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
			double x_uj = prediction(u, j);
			if (x_u_test > x_uj) {
				count_test ++;
			}
			if (x_u_val > x_uj) {
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

void model::AUC_coldItem(double* AUC_test, double* std, int* num_user)
{
	double* AUC_u_test = new double[nUsers];
	for (int u = 0; u < nUsers; u ++) {
		AUC_u_test[u] = -1;  // denote not testing
	}

	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		int item_test = test_per_user[u].first;
		int item_val  = val_per_user[u].first;
		
		if (pos_per_item[item_test].size() > 5) {
			continue;
		}

		double x_u_test = prediction(u, item_test);

		int count_test = 0;
		int max = 0;
		for (int j = 0; j < nItems; j ++) {
			if (pos_per_user[u].find(j) != pos_per_user[u].end() ||  // in training set
				item_test == j ||  // in test set
				item_val == j) {   // in val set 
				continue;
			}
			max ++;
			double x_uj = prediction(u, j);
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

void model::copyBestModel()
{
	for (int w = 0; w < NW; w ++) {
		bestW[w] = W[w];
	}
}

void model::saveModel(const char* path)
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
	fprintf(f, "}\n");
	fclose(f);

	fprintf(stderr, "\nModel saved to %s.\n", path);
}

/// model must be first initialized before calling this function
void model::loadModel(const char* path)
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
			break;
		}
	}
	in.close();
}

string model::toString()
{
	return "Empty Model!";
}