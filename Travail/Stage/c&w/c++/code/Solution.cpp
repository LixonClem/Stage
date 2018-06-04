#include "Solution.h"

Solution::Solution(PbData *pbdata)
{
	this->init(pbdata);
}
Solution::Solution(PbData *pbdata, double lambda, double mu, double nu)
{
	this->init(pbdata);
	this->parameters.lambda = lambda;
	this->parameters.mu = mu;
	this->parameters.nu = nu;
}
Solution::Solution(PbData *pbdata, vector<Route>& routes)
{
	this->init(pbdata);
	for(vector<Route>::iterator it = routes.begin(); it != routes.end(); ++it)
	{
		this->push_back(*it);
	}
}
Solution& Solution::operator = (const Solution& s)
{
	this->pbdata = s.pbdata;
	this->travelledDistance = s.travelledDistance;
	this->travelledTime = s.travelledTime;
	this->cost = s.cost;
	this->routes = s.routes;
	
	this->parameters = s.parameters;
	this->distancetobest = s.distancetobest;
	
	return *this;
}
void Solution::calculateDistances(Solution &S)
{
	calculateDistanceEdit(S);
	calculateDistanceExchange(S);
	calculateDistanceInsertion(S);
	calculateDistanceNbEqualArcs(S);
	calculateDistanceNbEqualEdges(S);
	calculateDistanceNbEqualRoute(S);
	calculateDistanceEdit(S);
	this->distancetobest.nbDifferentArcs = pbdata->getNbCustomers() + S.routes.size() - this->distancetobest.nbEqualArcs;
	this->distancetobest.nbDifferentEdges = pbdata->getNbCustomers() + S.routes.size() - this->distancetobest.nbEqualEdges;
}
void Solution::calculateDistanceEdit(Solution &S)
{
	vector<int> S1, S2;
	convertSolutionInVector(S1);
	convertSolutionInVector(S2, S);

	vector<vector<int>> d;

	for(int i=0; i<S1.size(); ++i)
	{
		vector<int> di;
		di.push_back(i);
		d.push_back(di);
	}

	for(int j=0; j<S2.size(); ++j)
	{
		d.at(0).push_back(j);
	}

	for(int i=1; i<S1.size(); ++i)
	{
		for(int j=1; j<S2.size(); ++j)
		{
			if(S1.at(i) == S2.at(j))
				d.at(i).push_back( d.at(i-1).at(j-1) );
			else
			{
				int dij = d.at(i-1).at(j) + 1;
				if(d.at(i).at(j-1) + 1 < dij) dij = d.at(i).at(j-1) + 1;
				if(d.at(i-1).at(j-1) + 1 < dij) dij = d.at(i-1).at(j-1) + 1;
				d.at(i).push_back( dij );
			}
		}
	}
	distancetobest.edit = d.at(S1.size() - 1).at(S2.size() - 1);
}
void Solution::calculateDistanceExchange(Solution &S)
{
	vector<int> S1, S2;
	convertSolutionInVector(S1);
	convertSolutionInVector(S2, S);

	int index,indexx;
	int n=0;
	bool zero;
	std::vector<int> indexStandby;

	/* should we reduce the length as much as possible ? */
	if(S1.size()<S2.size()){	/* how the size can be different if we normalize the vectors */
		while(S1.size()<S2.size())
			S1.push_back(0);				
	}

	if(S2.size()<S1.size()){
		while(S2.size()<S1.size())		
			S2.push_back(0);				
	}
	
	for (int i=1;i<S1.size()-1;++i) {	
		if(S1[i]!=S2[i]) {	
	
			if (indexStandby.size()!=0 && isPresent(S2[i],indexStandby,S1)) {
				index=getIndexfrom(S2[i],indexStandby,S1,0);
				swap(S1,i,index);
				++n;				
				if(S1[index]==S2[index]) {				
						indexx=getIndexfrom(index, indexStandby, 0);
						indexStandby.erase(indexStandby.begin()+indexx);
				}
			}
		
			else if (S2[i]==0) {
				zero=false;			
				for (int j=i+1;j<S1.size()-1;++j) {
					if (S1[j]==0 && S2[j]==S1[i] && zero==false) {					
						swap(S1,i,j);
						++n;
						zero=true;	
					}
				}				
				if (zero==false)				
					indexStandby.push_back(i);
			}
		
			else { 
				index=getIndexfrom(S2[i], S1, i+1); 
				swap(S1,i,index);
				++n;		
			}
		
		}		
	}
	
	distancetobest.exchange = n;
}
void Solution::calculateDistanceInsertion(Solution &S)
{
	vector<int> S1, S2;
	convertSolutionInVector(S1);
	convertSolutionInVector(S2, S);

	int length=S1.size()-1;
	std::vector<int> Ctemp(length);
	std::vector<std::vector<int> > C(length,Ctemp);

	for(int k=0; k<length; ++k){
		C[0][k]=0;
		C[k][0]=0;
	}

	for(int i=1;i<length;i++){
		for(int j=1;j<length;j++){
			if(S1[i]==S2[j]){
				C[i][j]=C[i-1][j-1]+1;
			}
			else{
				C[i][j]=std::max(C[i-1][j],C[i][j-1]);
			}
		}
	}

	distancetobest.insertion = length-1-C[length-1][length-1];
}
void Solution::calculateDistanceNbEqualArcs(Solution &S)
{
	int nbEqualsArcs = 0;
	vector<int> succS1, succS2;
	getSuccessorsVector(succS1);
	getSuccessorsVector(succS2, S);

	/* contribution to the arc distance of the arcs from customers to customer and from customers to the depot */
	for(int node=0; node<pbdata->getMaxId()+1; ++node)	/* succS1 and succS2 should have the same size since we are considering the same instance */
	{
		/* the -1 case is only to prevent instances where some ids are not considered */
		if(succS1.at(node) != -1 && succS1.at(node) == succS2.at(node))
			++nbEqualsArcs;
	}
	/* contribution to the arc distance of the arcs from the depot to the customers */
	/* since the normalisation of the solution we need only to compare the initial node of each route */
	int nbRoute = min(this->size(), S.size());
	for(int r=0; r<nbRoute; ++r)
	{
		if(this->routes.at(r).at(0).getId() == S.routes.at(r).at(0).getId())
			++nbEqualsArcs;
	}
	distancetobest.nbEqualArcs = nbEqualsArcs;
}
void Solution::calculateDistanceNbEqualEdges(Solution &S)
{
	int nbEqualsEdges = 0;
	vector<int> predS1, predS2, succS1, succS2;
	getSuccessorsVector(succS1);
	getSuccessorsVector(succS2, S);
	getPredecessorsVector(predS1);
	getPredecessorsVector(predS2, S);


	/* contribution to the edge distance of the arcs from customers to customer and from customers to the depot */
	for(int node=0; node<pbdata->getMaxId()+1; ++node)	/* succS1, succS2, predS1 and predS2 should have the same size since we are considering the same instance */
	{
		/* the -1 case is only to prevent instances where some ids are not considered */
		if(succS1.at(node) != -1 && (succS1.at(node) == succS2.at(node) || succS1.at(node) == predS2.at(node)) )
			++nbEqualsEdges;
	}
	/* contribution to the edge distance of the arcs from the depot to the customers */
	/* since the normalisation of the solution we need only to compare the initial node of each route */
	int nbRoute = min(this->size(), S.size());
	for(int r=0; r<nbRoute; ++r)
	{
		/* first part of the if condition: if the first node of the route is the same, we have the same arc in the solution, so the edge distance is incremented */
		/* second part of the if condition: if the first customer in route r in the first solution is last in one route of the second solution, */
		/* namely its successor is 0, then we have the opposite arc in the two solution, and this contributes to the distance  */
		if((this->routes.at(r).at(0).getId() == S.routes.at(r).at(0).getId()) || (succS2.at(this->routes.at(r).at(0).getId()) == 0))
			++nbEqualsEdges;
	}
	distancetobest.nbEqualEdges = nbEqualsEdges;
}
void Solution::calculateDistanceNbEqualRoute(Solution &S)
{
	int nbRoute = min(this->size(), S.size());
	int nbEqualRoutes = 0;
	for(int r=0; r<nbRoute; ++r)
	{
		if(this->getRoute(r).size() == S.getRoute(r).size())
		{
			int pos = 0;
			bool equal = true;
			do
			{
				if(this->getRoute(r).at(pos) != S.getRoute(r).at(pos))
					equal = false;
				++pos;
			}while(equal && pos < this->getRoute(r).size());

			if(equal) ++nbEqualRoutes;
		}
	}
	distancetobest.nbEqualRoutes = nbEqualRoutes;
}
bool Solution::control()
{
	bool check = true;
	vector<int> customers;
	for(int i=0; i<=pbdata->getMaxId(); ++i)
		customers.push_back(0);
	
	for(int r=0; r < this->routes.size(); ++r)
	{
		for(int n=0; n<this->routes.at(r).size(); ++n)
		{
			customers.at(this->routes.at(r).at(n).getId()) = 1;
		}
	}
	for(int i=pbdata->getMinId(); i<=pbdata->getMaxId(); ++i)
	{
		if(customers.at(i) == 0)
		{
			if(pbdata->getNode(i).isCustomer())
			{
				cout << "missing a customer: " << i << endl;
				check = false;
			}
		}
	}
	double distance = 0;
	for(int r=0; r < this->routes.size(); ++r)
	{
		distance += pbdata->getTravelledDistance(pbdata->getDepot(), this->routes.at(r).at(0));
		for(int i=0; i<this->routes.at(r).size() - 1; ++i)
		{
			distance += pbdata->getTravelledDistance(this->routes.at(r).at(i), this->routes.at(r).at(i + 1));
		}
		distance += pbdata->getTravelledDistance(this->routes.at(r).at(this->routes.at(r).size() - 1), pbdata->getDepot());
	}
	if(abs(distance - this->getTravelledDistance()) > 0.0001)
	{
		cout << "distance is wrong\n";
		check = false;
	}
	
	for(int r=0; r < this->routes.size(); ++r)
	{
		double q = 0;
		for(int i=0; i<this->routes.at(r).size(); ++i)
		{
			q += this->routes.at(r).at(i).getQuantity();
		}
		if(q > pbdata->getVehicleCapacity())
		{
			check = false;
			cout << "route violates capacity\n";
		}
		
	}
	
	return check;
}
void Solution::convertSolutionInVector(vector<int> &v)
{
	for(int r=0; r<this->size(); ++r)
	{
		v.push_back(0);
		for(int i=0; i<this->routes.at(r).size(); ++i)
		{
			v.push_back(this->routes.at(r).at(i).getId());
		}
		v.push_back(0);
	}
	/* we want all the string of the same size, then we complete with as many zero as needed */
	for(int i=0; i<pbdata->getNbCustomers() + 1 - this->size(); ++i)
	{
		v.push_back(0);
	}
}
void Solution::convertSolutionInVector(vector<int> &v, Solution &S)
{
	for(int r=0; r<S.size(); ++r)
	{
		v.push_back(0);
		for(int i=0; i<S.getRouteSize(r); ++i)
		{
			v.push_back(S.getRoute(r).at(i).getId());
		}
		v.push_back(0);
	}
	/* we want all the string of the same size, then we complete with as many zero as needed */
	for(int i=0; i<pbdata->getNbCustomers() + 1 - S.size(); ++i)
	{
		v.push_back(0);
	}
}
double Solution::getCost()
{
	return this->travelledDistance;
}
int Solution::getIndexfrom(int k, std::vector<int>& index, std::vector<int>& S, int ifrom)
{
	assert(ifrom<index.size());
	for (int i=ifrom;i<index.size();++i){
		if (S[index[i]]==k) return index[i];	
	}
}
//
//int Solution::getIndexfrom(int k, std::vector<int>& S, int ifrom)
//{
//	assert (ifrom<=S.size()-1);
//	for (int i=ifrom;i<=S.size()-1;++i)
//	{
//		if (S[i]==k) return i;	
//	}
//}

void Solution::getPredecessorsVector(vector<int>& pred)
{
	for(int v=0; v<=pbdata->getMaxId(); ++v)
	{
		pred.push_back(-1);	/* initialisation with a -1 for each id */
	}

	for(int r=0; r<this->size(); ++r)
	{
		pred.at(this->routes.at(r).at( 0 ).getId()) = 0;	/* the predecessor of the first is the depot */
		for(int v=1; v<this->routes.at(r).size(); ++v)
		{
			pred.at(this->routes.at(r).at(v).getId()) = this->routes.at(r).at(v-1).getId();
		}
	}
}
void Solution::getPredecessorsVector(vector<int>& pred, Solution &S)
{
	for(int v=0; v<=pbdata->getMaxId(); ++v)
	{
		pred.push_back(-1);	/* initialisation with a -1 for each id */
	}

	for(int r=0; r<S.size(); ++r)
	{
		pred.at(S.routes.at(r).at( 0 ).getId()) = 0;		/* the predecessor of the first is the depot */
		for(int v=1; v<S.routes.at(r).size(); ++v)
		{
			pred.at(S.routes.at(r).at(v).getId()) = S.routes.at(r).at(v-1).getId();
		}
	}
}
void Solution::getSuccessorsVector(vector<int>& succ)
{
	for(int v=0; v<=pbdata->getMaxId(); ++v)
	{
		succ.push_back(-1);	/* initialisation with a -1 for each id */
	}

	for(int r=0; r<this->size(); ++r)
	{
		for(int v=0; v<this->routes.at(r).size() - 1; ++v)
		{
			succ.at(this->routes.at(r).at(v).getId()) = this->routes.at(r).at(v+1).getId();
		}
		succ.at(this->routes.at(r).at( this->routes.at(r).size() - 1 ).getId()) = 0;	/* the successor of the last is the depot */
	}
}
void Solution::getSuccessorsVector(vector<int>& succ, Solution &S)
{
	for(int v=0; v<=pbdata->getMaxId(); ++v)
	{
		succ.push_back(-1);	/* initialisation with a -1 for each id */
	}

	for(int r=0; r<S.size(); ++r)
	{
		for(int v=0; v<S.routes.at(r).size() - 1; ++v)
		{
			succ.at(S.routes.at(r).at(v).getId()) = S.routes.at(r).at(v+1).getId();
		}
		succ.at(S.routes.at(r).at( S.routes.at(r).size() - 1 ).getId()) = 0;	/* the successor of the last is the depot */
	}
}
Route Solution::getRoute(int r)
{
	return this->routes.at(r);
}
int Solution::getRouteSize(int r)
{
	return this->routes.at(r).size();
}
double Solution::getTravelledTime()
{
	return this->travelledTime;
}
double Solution::getTravelledDistance()
{
	return this->travelledDistance;
}
bool Solution::handCheck()
{
	Route R1(this->pbdata);
	R1.push_back(pbdata->getNode(6)); R1.push_back(pbdata->getNode(26)); R1.push_back(pbdata->getNode(11)); R1.push_back(pbdata->getNode(16)); 
	R1.push_back(pbdata->getNode(10)); R1.push_back(pbdata->getNode(23)); R1.push_back(pbdata->getNode(30)); R1.push_back(pbdata->getNode(21)); 

	Route R2(this->pbdata);
	R2.push_back(pbdata->getNode(7)); R2.push_back(pbdata->getNode(22)); R2.push_back(pbdata->getNode(18)); R2.push_back(pbdata->getNode(20)); 
	R2.push_back(pbdata->getNode(32)); R2.push_back(pbdata->getNode(14));

	Route R3(this->pbdata);
	R3.push_back(pbdata->getNode(15)); R3.push_back(pbdata->getNode(19)); R3.push_back(pbdata->getNode(9)); R3.push_back(pbdata->getNode(5)); 
	R3.push_back(pbdata->getNode(12)); R3.push_back(pbdata->getNode(29)); R3.push_back(pbdata->getNode(24)); R3.push_back(pbdata->getNode(3));
	R3.push_back(pbdata->getNode(4)); R3.push_back(pbdata->getNode(27)); 

	Route R4(this->pbdata);
	R4.push_back(pbdata->getNode(17)); R4.push_back(pbdata->getNode(8)); R4.push_back(pbdata->getNode(2)); 
	R4.push_back(pbdata->getNode(13)); R4.push_back(pbdata->getNode(31)); 

	Route R5(this->pbdata);
	R5.push_back(pbdata->getNode(25)); R5.push_back(pbdata->getNode(28)); 

	this->push_back(R1); this->push_back(R2); this->push_back(R3);
	this->push_back(R4); this->push_back(R5);

	return this->control();
}
void Solution::init(PbData *pbdata)
{
	this->pbdata = pbdata;
	this->travelledDistance = 0;
	this->travelledTime = 0;
	this->cost = 0;
}
bool Solution::isPresent(int k, std::vector<int>& index, std::vector<int>& S)
{
	for (int i=0;i<index.size();++i){
	
		if (S[index[i]]==k) return true;	
	
	}
	return false;
}
void Solution::normalize()
{
	for(int r=0; r<this->routes.size(); ++r){
		if(this->routes.at(r).getFirst().getId() > this->routes.at(r).getLast().getId())
			this->routes.at(r).reverse();
	}
	std::sort(this->routes.begin(), this->routes.end());
}
void Solution::push_back(Route &route)
{
	this->routes.push_back(route);
	this->travelledDistance += route.getTravelledDistance();
	this->travelledTime += route.getTravelledTime();
	this->cost += route.getTravelledDistance();
}
int Solution::size()
{
	return this->routes.size();
}
string Solution::toString()
{
	string str = "distance exchange: " + to_string(this->distancetobest.exchange);
	str += "\ndistance inseertion: " + to_string(this->distancetobest.insertion);
	str += "\nnb different arcs: " + to_string(this->distancetobest.nbDifferentArcs);
	str += "\nnb different edges: " + to_string(this->distancetobest.nbDifferentEdges);
	str += "\nnb equal arcs: " + to_string(this->distancetobest.nbEqualArcs);
	str += "\nnb equal edges: " + to_string(this->distancetobest.nbEqualEdges);
	str += "\nnb equal routes: " + to_string(this->distancetobest.nbEqualRoutes);
	return str;
}
void Solution::swap(std::vector<int>& S, int i, int j) 
{
	int var;
	var=S[i];
	S[i]=S[j];
	S[j]=var;
}
void Solution::write()
{
	ofstream mystream(pbdata->getInstanceName() + "analysis.dat", ios::app);
	//this->writeHeader(mystream);
	this->writeParamters(mystream);
	this->writeDistances(mystream);
	this->writeSerialized(mystream);
	mystream.close();
}
void Solution::writeHeader(ofstream &mystream)
{
	mystream << "l\tm\tn\tcost\ted\tex\tin\teA\teE\teR\tdA\tdE"	<< endl;
}
void Solution::writeDistances(ofstream &mystream)
{
	mystream << this->distancetobest.edit << "\t";
	mystream << this->distancetobest.exchange << "\t";
	mystream << this->distancetobest.insertion << "\t";
	mystream << this->distancetobest.nbEqualArcs << "\t";
	mystream << this->distancetobest.nbEqualEdges << "\t";
	mystream << this->distancetobest.nbEqualRoutes << "\t";
	mystream << this->distancetobest.nbDifferentArcs << "\t";
	mystream << this->distancetobest.nbDifferentEdges << "\t";
}
void Solution::writeParamters(ofstream &mystream)
{
	mystream << this->parameters.lambda << "\t" << this->parameters.mu << "\t" << this->parameters.nu << "\t" << this->travelledDistance << "\t";
}
void Solution::writeSerialized(ofstream &mystream)
{
	/* the solution is written as 0 route1 0 route2 0 route3 0 ... 0 routeN 0 */
	/* the longest string has size n + n + 1 (with all round-trips) */
	for(int r=0; r<this->size(); ++r)
	{
		mystream << 0 << "\t";
		for(int i=0; i<this->routes.at(r).size(); ++i)
		{
			mystream << this->routes.at(r).at(i).getId() << "\t";
		}
	}
	/* we want all the string of the same size, then we complete with as many zero as needed */
	for(int i=0; i<pbdata->getNbCustomers() + 1 - this->size(); ++i)
	{
		mystream << 0 << "\t";
	}
	mystream << endl;
}
void Solution::writeGnuplot()
{
	/* head of the plot */
	ofstream mystream(pbdata->getGnuplotPath() + pbdata->getInstanceName() + ".dat");
	mystream << "reset\n";
	mystream << "set autoscale\n";
	mystream << "set size square\n";
	mystream << "set output '" << pbdata->getGnuplotPath() + pbdata->getInstanceName() << ".pdf'\n";
	mystream << "set terminal pdf enhanced\n";
	mystream << "plot ";
	
	for(int route=0; route < this->routes.size(); ++route)
	{
		/* points in route route */
		ofstream myCoordStream(pbdata->getGnuplotPath() + pbdata->getInstanceName() + "_route" + to_string(route) + "_coords.dat");
		myCoordStream << pbdata->getDepot().getX() << "\t" << pbdata->getDepot().getY() << endl;
		for(int i=0; i<this->routes.at(route).size(); ++i)
		{
			myCoordStream << this->routes.at(route).getNodes().at(i).getX() << "\t" << this->routes.at(route).getNodes().at(i).getY() << endl;
		}
		myCoordStream << pbdata->getDepot().getX() << "\t" << pbdata->getDepot().getY() << endl;
		myCoordStream.close();
		mystream << "'" + pbdata->getGnuplotPath() + pbdata->getInstanceName() + "_route" + to_string(route) + "_coords.dat' with linespoints ls " + to_string(route) + " notitle,\\\n";
	}
	
	/* footer of the plot */
	
	mystream.close();
}