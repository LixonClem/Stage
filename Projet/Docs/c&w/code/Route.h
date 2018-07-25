#pragma once

#include "Node.h"
#include "PbData.h"

using namespace std;

class Route
{
	/* attributes */
	PbData *pbdata;
	vector<Node> nodes;	/* it contains the nodes that are visited, not the depot */
	double quantity;
	double travelledTime;
	double travelledDistance;

public:

	/* constructors */
	Route();
	Route(PbData *pbdata);
	Route(PbData *pbdata, Node customer);

	/* operators */
	Route& operator = (const Route& r);
	friend bool operator < (Route& R1, Route& R2);

	Node at(int i);
	bool control();
	double getQuantity();
	double getTravelledTime();
	double getTravelledDistance();
	vector<Node> getNodes();
	Node& getFirst();
	Node& getLast();

	void push_back(Node &node);
	void reverse();
	int size();
	void fusion(Route& R);

private:
	void init(PbData *pbdata);
};
