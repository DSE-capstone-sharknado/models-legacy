#include "POP.hpp"

double POP::prediction(int user, int item)
{
	return (double)pos_per_item[item].size();	// return popularity of item
}

string POP::toString()
{
	return "POP";
}