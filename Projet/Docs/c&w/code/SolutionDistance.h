#pragma once
#include <iostream>
#include <string>

#include "Common.h"

using namespace std;

/* To respect the format of VRP-REP we introduce the class node with an attribute id and type */
/* type = 0 correspondds to the depot, while type = 1 corresponds to the customers */
/* in the VRP-REP format the depot does not necesserily have id = 0, but can have id = 1 */
/* on the other hand, since the depot is usually put as first it should (but not must) be */
/* stored in position 0 in the vector nodes  */

class SolutionDistance
{
	/* attributes */
	double lambda;
	double mu;
	double nu;

	int distanceEdit;
	int distanceExchange;
	int distanceInsertion;
	int distanceNbEqualArcs;
	int distanceNbEqualEdges;
	int distanceNbEqualRoutes;

public:
	
	/* constructors */
	SolutionDistance();
	
};
