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

class Node
{
	/* attributes */
	int id;
	int type;
	int quantity;
	int serviceTime;
	double x;
	double y;

public:

	/* operators */
	Node& operator = (const Node& n);
	friend int operator < (const Node& n1, const Node& n2);
	bool Node::operator==(const Node& n);
	bool Node::operator!=(const Node& n);

	/* constructors */
	Node(int id, int type, double x, double y);
	Node(int id, int type, double x, double y, int serviceTime, int quantity);
	Node(const Node& node);
	Node();

	/* getters */
	int getId();
	int getQuantity();
	int getServiceTime();
	int getType();
	double getX();
	double getY();

	/* setters */
	void setQuantity(double quantity);
	void setServiceTime(double serviceTime);
	
	/* methods */
	bool isCustomer();
	bool isDepot();
	string toString();
};
