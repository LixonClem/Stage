#include "Solver.h"

Solver::Solver(PbData* pbdata)
{
	this->pbdata = pbdata;
}
void Solver::clarkeAndWright(Solution &solution, double lambda, double mu, double nu)
{
	double maxSaving = 0;
	vector<Route> routes;
	vector<Route>::iterator rFusion1, rFusion2;
	initialiseSingleCustomersRoutes(routes);
	calculateSavings(routes, rFusion1, rFusion2, maxSaving, lambda, mu, nu);
	
	while(maxSaving > 0)
	{
		rFusion1->fusion(*rFusion2);
		routes.erase(rFusion2);
		maxSaving = 0;
		calculateSavings(routes, rFusion1, rFusion2, maxSaving, lambda, mu, nu);
	}

	for(vector<Route>::iterator it = routes.begin(); it != routes.end(); ++it)
	{
		solution.push_back(*it);
	}
	//solution.control();
	solution.normalize();
}
void Solver::clarkeAndWrightRnd(Solution &solution, double lambda, double mu, double nu)
{
	double maxSaving = 0;
	vector<Route> routes;
	vector<Route>::iterator rFusion1, rFusion2;

	initialiseSingleCustomersRoutes(routes);
	calculateSavingsRnd(routes, rFusion1, rFusion2, maxSaving, lambda, mu, nu);
	
	while(maxSaving > 0)
	{
		rFusion1->fusion(*rFusion2);
		routes.erase(rFusion2);
		maxSaving = 0;
		calculateSavingsRnd(routes, rFusion1, rFusion2, maxSaving, lambda, mu, nu);
	}

	for(vector<Route>::iterator it = routes.begin(); it != routes.end(); ++it)
	{
		solution.push_back(*it);
	}
	//solution.control();
	solution.normalize();
}
void Solver::initialiseSingleCustomersRoutes(vector<Route>& routes)
{
	map<int, Node>::iterator it1;
	for(int i=1; i<=pbdata->getNbNodes(); ++i)	/* that's because of the map structure  */
	{
		Node node = pbdata->getNode(i);
		if(node.isCustomer())
		{
			Route R(pbdata, node);
			routes.push_back(R);
		}
	}
}
void Solver::calculateSavings(vector<Route>& routes, vector<Route>::iterator& itBest1, vector<Route>::iterator& itBest2, double& maxSaving, double lambda, double mu, double nu)
{
	vector<Route>::iterator it1, it2;
	for(it1 = routes.begin(); it1 != routes.end(); ++it1)
	{
		for(it2 = routes.begin(); it2 != routes.end(); ++it2)
		{
			if(it2 != it1 && it1->getQuantity() + it2->getQuantity() <= pbdata->getVehicleCapacity())
			{
				double saving = pbdata->getTravelledDistance(it1->getLast(), pbdata->getDepot()) 
				+ pbdata->getTravelledDistance(pbdata->getDepot(), it2->getFirst())
				- lambda * pbdata->getTravelledDistance(it1->getLast(), it2->getFirst())
				+ mu * abs( pbdata->getTravelledDistance(pbdata->getDepot(), it1->getLast()) - pbdata->getTravelledDistance(it2->getFirst(), pbdata->getDepot())) 
				+ nu * ( it1->getLast().getQuantity() + it2->getFirst().getQuantity() ) / this->pbdata->getAverageCustomerDemand();

				if(saving > maxSaving)
				{
					maxSaving = saving;
					itBest1 = it1; itBest2 = it2;
				}
			}
		}
	}
}
void Solver::calculateSavingsRnd(vector<Route>& routes, vector<Route>::iterator& itBest1, vector<Route>::iterator& itBest2, double& maxSaving, double lambda, double mu, double nu)
{
	vector<Route>::iterator it1, it2;
	vector<vector<Route>::iterator> vectorItBest1, vectorItBest2;
	for(it1 = routes.begin(); it1 != routes.end(); ++it1)
	{
		for(it2 = routes.begin(); it2 != routes.end(); ++it2)
		{
			if(it2 != it1 && it1->getQuantity() + it2->getQuantity() <= pbdata->getVehicleCapacity())
			{
				double saving = pbdata->getTravelledDistance(it1->getLast(), pbdata->getDepot()) 
				+ pbdata->getTravelledDistance(pbdata->getDepot(), it2->getFirst())
				- lambda * pbdata->getTravelledDistance(it1->getLast(), it2->getFirst())
				+ mu * abs( pbdata->getTravelledDistance(pbdata->getDepot(), it1->getLast()) - pbdata->getTravelledDistance(it2->getFirst(), pbdata->getDepot())) 
				+ nu * ( it1->getLast().getQuantity() + it2->getFirst().getQuantity() ) / this->pbdata->getAverageCustomerDemand();

				if(saving > maxSaving)
				{
					/* new best is found. Memory is erased and new best saving routes are saved */
					vectorItBest1.erase(vectorItBest1.begin(), vectorItBest1.end());
					vectorItBest2.erase(vectorItBest2.begin(), vectorItBest2.end());
					maxSaving = saving;
					vectorItBest1.push_back(it1);
					vectorItBest2.push_back(it2);
				}
				else if(abs(saving - maxSaving) < 0.001)
				{
					/* equivalent best is found and stored in memory */
					vectorItBest1.push_back(it1);
					vectorItBest2.push_back(it2);
				}
			}
		}
	}

	if(maxSaving > 0)
	{
		int posRnd = rand() % vectorItBest1.size();
		cout << "size of best saving: " << vectorItBest1.size() << endl;
		itBest1 = vectorItBest1.at(posRnd); itBest2 = vectorItBest2.at(posRnd);
	}

}