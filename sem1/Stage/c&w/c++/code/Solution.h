#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include "PbData.h"
#include "Route.h"


using namespace std;

struct parameters{
	struct parameters& operator = (const parameters& p)
		{lambda = p.lambda; mu = p.mu; nu = p.nu; return *this;}
	double lambda;
	double mu;
	double nu;
};
struct distancesolution{
	struct distancesolution& operator = (const distancesolution& d)
		{edit = d.edit; exchange = d.exchange; insertion= d.insertion; nbEqualArcs = d.nbEqualArcs; nbEqualEdges = d.nbEqualEdges; 
		nbEqualRoutes = d.nbEqualRoutes; nbDifferentArcs = d.nbDifferentArcs; nbDifferentEdges = d.nbDifferentEdges;
		return *this;}
	int edit;
	int exchange;
	int insertion;
	int nbEqualArcs;
	int nbEqualEdges;
	int nbEqualRoutes;
	int nbDifferentArcs;
	int nbDifferentEdges;
};


class Solution
{
	/* attributes */
	PbData *pbdata;
	vector<Route> routes;
	double cost;
	double travelledTime;
	double travelledDistance;
	parameters parameters;
	distancesolution distancetobest;

public:

	/* operators */
	Solution& operator = (const Solution& s);

	/* constructors */
	Solution();
	Solution(PbData *pbdata);
	Solution(PbData *pbdata, double lambda, double mu, double nu);
	Solution(PbData *pbdata, vector<Route>& routes);

	bool control();
	void calculateDistances(Solution &S);
	void calculateDistanceEdit(Solution &S);
	void calculateDistanceExchange(Solution &S);
	void calculateDistanceInsertion(Solution &S);
	void calculateDistanceNbEqualArcs(Solution &S);
	void calculateDistanceNbEqualEdges(Solution &S);
	void calculateDistanceNbDifferentArcs(Solution &S);
	void calculateDistanceNbDifferentEdges(Solution &S);
	void calculateDistanceNbEqualRoute(Solution &S);
	double getCost();
	Route getRoute(int r);
	int getRouteSize(int r);
	double getTravelledTime();
	double getTravelledDistance();
	bool handCheck();
	bool isPresent(int k, std::vector<int>& index, std::vector<int>& S);
	void normalize();
	void push_back(Route &route);
	int size();
	void swap(std::vector<int>& S, int i, int j);
	string toString();
	void write();
	void writeGnuplot();
	
private:
	void convertSolutionInVector(vector<int> &v);
	void convertSolutionInVector(vector<int> &v, Solution &S);
	int getIndexfrom(int k, std::vector<int>& S, int ifrom);
	int getIndexfrom(int k, std::vector<int>& index, std::vector<int>& S, int ifrom);
	void getPredecessorsVector(vector<int>& succ);
	void getPredecessorsVector(vector<int>& succ, Solution &S);
	void getSuccessorsVector(vector<int>& pred);
	void getSuccessorsVector(vector<int>& pred, Solution &S);
	void init(PbData *pbdata);
	
	void writeDistances(ofstream &mystream);
	void writeHeader(ofstream &mystream);
	void writeParamters(ofstream &mystream);
	void writeSerialized(ofstream &mystream);

};

