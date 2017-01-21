#pragma once

#include "model.hpp"

class POP : public model
{
public:
	POP(corpus* corp) : model(corp) {}
	~POP(){}

	double prediction(int user, int item);
	string toString();
};
