#include "Point.h"

Point& Point::operator = (const Point& P)
{
	this->x = P.x;
	this->y = P.y;
	this->id = P.id;
	return *this;
}
Point::Point() : x(0), y(0), id(0)
{
}
Point::Point(int mId, int mX, int mY) : id(mId), x(mX), y(mY)
{
}
double Point::getCostTo(int toPoint)
{
	return this->getDistanceTo(toPoint);
}
int Point::getX()
{
	return this->x;
}
int Point::getY()
{
	return this->y;
}
int Point::getId()
{
	return this->id;
}
double Point::getDistanceTo(int toPoint)
{
	return this->distance.at(toPoint);
}
double Point::getReplenishmentCostTo(int toPoint)
{
	return this->getReplenishmentDistanceTo(toPoint);
}
double Point::getReplenishmentDistanceTo(int toPoint)
{
	return this->replenishmentDistance.at(toPoint);
}
double Point::getReplenishmentTravelTimeTo(int toPoint)
{
	return this->replenishmentTravelTime.at(toPoint);
}
double Point::getTravelTimeTo(int toPoint)
{
	return this->travelTime.at(toPoint);
}
void Point::calculateDistance(Point &p)
{
	double dist = sqrt( pow(this->x - p.x, 2) + pow(this->y - p.y, 2));
	dist = dist * 10; dist = (int)dist; dist = dist / 10;
	distance.push_back(dist);
}
void Point::calculateReplenishmentDistance(Point &Depot, Point &p)
{
	replenishmentDistance.push_back(this->getDistanceTo(Depot.getId()) + Depot.getDistanceTo(p.getId()));
}
void Point::calculateReplenishmentTravelTime(Point &Depot, Point &p)
{
	replenishmentTravelTime.push_back(this->getTravelTimeTo(Depot.getId()) + Depot.getTravelTimeTo(p.getId()));
}
void Point::calculateTravelTime(Point &p)
{
	double time = sqrt( pow(this->x - p.x, 2) + pow(this->y - p.y, 2));
	time = time * 10; time = (int)time; time = time / 10;
	travelTime.push_back(time);
}
string Point::toString() 
{
	string str = "id:" + to_string(this->id) + " (" + to_string(this->x) + ", " + to_string(this->y) + ")\n";
	for(int i=0; i<travelTime.size(); ++i) str += "\t distTo(" + to_string(i) + "): " + to_string(travelTime.at(i)) + "\n";
	return str;
}
