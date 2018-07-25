#pragma once

#include "Node.h"
#include "PbData.h"
#include "Route.h"
#include "Solution.h"

using namespace std;

class Solver
{
	/* attributes */
	PbData *pbdata;
	
public:

	/* constructors */
	Solver(PbData* pbdata);
	
	void clarkeAndWright(Solution &solution, double lambda, double mu, double nu);
	void clarkeAndWrightRnd(Solution &solution, double lambda, double mu, double nu);

private:

	void initialiseSingleCustomersRoutes(vector<Route>& routes);
	void calculateSavings(vector<Route>& routes, vector<Route>::iterator& it1, vector<Route>::iterator& it2, double& maxSaving, double lambda, double mu, double nu);
	void calculateSavingsRnd(vector<Route>& routes, vector<Route>::iterator& it1, vector<Route>::iterator& it2, double& maxSaving, double lambda, double mu, double nu);
};
