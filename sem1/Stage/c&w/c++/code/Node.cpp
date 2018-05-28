#include "Node.h"

Node& Node::operator = (const Node& node)
{
	this->id = node.id;
	this->type = node.type;
	this->quantity = node.quantity;
	this->serviceTime = node.serviceTime;
	this->x = node.x;
	this->y = node.y;
	return *this;
}
int operator < (const Node& n1, const Node& n2)
{
	cout << "ordering operator still to be coded!!\n";
	system("pause");
	return 0;
}
bool Node::operator==(const Node& n)
{
	return this->id == n.id;
}
bool Node::operator!=(const Node& n)
{
	return !(*this == n);
}
Node::Node()
{
	this->id = -1;
	this->type = -1;
	this->quantity = 0;
	this->serviceTime = 0;
	this->x = 0;
	this->y = 0;
}
Node::Node(int id, int type, double x, double y) : 
	id(id), type(type), x(x), y(y), serviceTime(0), quantity(0)
{
}
Node::Node(int id, int type, double x, double y, int serviceTime, int quantity) : 
	id(id), type(type), x(x), y(y), quantity(quantity), serviceTime(serviceTime)
{
}
Node::Node(const Node& node)
{
	this->id = node.id;
	this->type = node.type;
	this->quantity = node.quantity;
	this->serviceTime = node.serviceTime;
	this->x = node.x;
	this->y = node.y;
}
int Node::getId()
{
	return this->id;
}
double Node::getX()
{
	return this->x;
}
double Node::getY()
{
	return this->y;
}
int Node::getServiceTime()
{
	return this->serviceTime;
}
int Node::getType()
{
	return this->type;
}
int Node::getQuantity()
{
	return this->quantity;
}
bool Node::isCustomer()
{
	return (type == 1);
}
bool Node::isDepot()
{
	return (type == 0);
}
void Node::setQuantity(double quantity)
{
	this->quantity = quantity;
}
void Node::setServiceTime(double serviceTime)
{
	this->serviceTime = serviceTime;
}
string Node::toString()
{
	return "id: " + to_string(this->id) + "\ttype: " + to_string(this->type) + "\tx: " + to_string(this->x) + "\ty: " + to_string(this->y) + "\tquantity: " + to_string(this->quantity) + "\tserviceTime: " + to_string(this->serviceTime);
}