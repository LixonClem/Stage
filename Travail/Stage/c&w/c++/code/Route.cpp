#include "Route.h"

Route::Route()
{
}
Route::Route(PbData *pbdata)
{
	init(pbdata);
}
Route::Route(PbData *pbdata, Node customer)
{
	init(pbdata);
	this->push_back(customer);
}
bool operator < (Route& R1, Route& R2)
{
	return R1.getFirst().getId() < R2.getFirst().getId();
}
Route& Route::operator = (const Route& r)
{
	this->pbdata = r.pbdata;
	this->nodes = r.nodes;
	this->quantity = r.quantity;
	this->travelledTime = r.travelledTime;
	this->travelledDistance = r.travelledDistance;
	return *this;
}
Node Route::at(int i)
{
	return this->nodes.at(i);
}
bool Route::control()
{
	double dist = 0, quantity = 0;
	dist += pbdata->getTravelledDistance(pbdata->getDepot(), this->at(0));
	for(int n=1; n<this->size(); ++n)
	{
		dist += pbdata->getTravelledDistance(this->at(n-1), this->at(n));
	}
	dist += pbdata->getTravelledDistance(this->at(this->size() - 1), pbdata->getDepot());

	if(abs(dist - this->travelledDistance) > 0.01)
	{
		system("error in control");
		return false;
	}
}
double Route::getQuantity()
{
	return this->quantity;
}
double Route::getTravelledTime()
{
	return this->travelledTime;
}
double Route::getTravelledDistance()
{
	return this->travelledDistance;
}
Node& Route::getFirst()
{
	return this->nodes.at(0);
}
Node& Route::getLast()
{
	return this->nodes.at(this->nodes.size() - 1);
}
vector<Node> Route::getNodes()
{
	return this->nodes;
}
void Route::fusion(Route& R)
{
	/* the route R is appended to the this route */
	this->quantity += R.quantity;
	this->travelledDistance += pbdata->getTravelledDistance(this->getLast(), R.getFirst()) 
		+ R.getTravelledDistance()
		- pbdata->getTravelledDistance(this->getLast(), pbdata->getDepot()) 
		- pbdata->getTravelledDistance(pbdata->getDepot(), R.getFirst());

	this->travelledTime += pbdata->getTravelledTime(this->getLast(), R.getFirst()) 
		+ R.getTravelledTime()
		- pbdata->getTravelledTime(this->getLast(), pbdata->getDepot()) 
		- pbdata->getTravelledTime(pbdata->getDepot(), R.getFirst());

	this->nodes.insert(this->nodes.end(), R.nodes.begin(), R.nodes.end());
}
void Route::init(PbData *pbdata)
{
	this->pbdata = pbdata;
	this->quantity = 0;
	this->travelledTime = 0;
	this->travelledDistance = 0;
}
void Route::push_back(Node &node)
{
	this->quantity += node.getQuantity();

	if(this->nodes.size() == 0)
	{
		this->travelledDistance = pbdata->getTravelledDistance(pbdata->getDepot(), node) + pbdata->getTravelledDistance(node, pbdata->getDepot());
		this->travelledTime = pbdata->getTravelledTime(pbdata->getDepot(), node) + pbdata->getTravelledTime(node, pbdata->getDepot());
	}
	else
	{
		Node& last(this->nodes.at(this->nodes.size() - 1));
		this->travelledDistance += pbdata->getTravelledDistance(last, node) + pbdata->getTravelledDistance(node, pbdata->getDepot()) - pbdata->getTravelledDistance(last, pbdata->getDepot());
		this->travelledTime += pbdata->getTravelledTime(last, node) + pbdata->getTravelledTime(node, pbdata->getDepot()) - pbdata->getTravelledTime(last, pbdata->getDepot());
	}
	this->nodes.push_back(node);
}
void Route::reverse()
{
	std::reverse(this->nodes.begin(), this->nodes.end());
}
int Route::size()
{
	return this->nodes.size();
}
