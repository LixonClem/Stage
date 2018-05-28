#pragma once
#include <math.h>
#include <string>
#include <vector>
#include "Common.h"

using namespace std;

/* Travel time, travel distances and cost are equal in this vesion of the problem		*/
/* The code can memorize two different matrices for times and distances in case of need */
/* of solving this kind of instances. The getcost methods return the distance			*/
/* These methods are only present for generality popruses. In case of need more			*/
/* complicated cost functions can be easily handled										*/

class Point
{
	/* attributes */
	int id;
	int x;
	int y;
	vector<double> distance;
	vector<double> replenishmentDistance;
	vector<double> replenishmentTravelTime;
	vector<double> travelTime;

public:

	/* operators */
	Point& operator = (const Point& P);		//Point& Point::operator = (const Point& P);		//http://stackoverflow.com/questions/5642367/extra-qualification-error-in-c

	/* constructors */
	Point();
	Point(int mId, int mX, int mY);

	/* getters */
	double getCostTo(int toPoint);
	double getDistanceTo(int toPoint);
	int getId();
	int getX();
	int getY();
	double getReplenishmentCostTo(int toPoint);
	double getReplenishmentDistanceTo(int toPoint);
	double getReplenishmentTravelTimeTo(int toPoint);
	double getTravelTimeTo(int toPoint);

	/* methods */
	void calculateDistance(Point &p);								/* travel times and distances are equal in our case but better be general: on sait jamais  */
	void calculateReplenishmentDistance(Point & Depot, Point &p);
	void calculateReplenishmentTravelTime(Point & Depot, Point &p);
	void calculateTravelTime(Point &p);	
	string toString();

};
