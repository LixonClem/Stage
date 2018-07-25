#pragma once
#include <algorithm>	// to call the sort procedure on a vector
#include <fstream>		// to be able to read external files
#include <iostream>		// to be able to print messages in the console
#include <vector>		// to use the vector structure
#include <map>			// to use the vector structure
#include <cmath>

#include "Node.h"
#include "Common.h"

using namespace std;	

class PbData
{
	/* user data */
	Common::user myuser;
	string gnuplotPath;
	string instancePath;
	string instanceOut;

	/* instance naming */
	string dataSet;
	string instanceName;

	/* configuration parameters */

	/* instance parameters */
	double averageCustomerDemand;
	int nbVehicles;
	int vehicleCapacity;
	int maxId;
	int minId;
	Node depot;
	map<int, Node> nodes;
	map<int, map<int, double>> travelledDistance;
	map<int, map<int, double>> travelledTime;
		
public:

	/* constructors */
	PbData(Common::user myuser, string instanceName);
	
	/* getters */
	Node & getNode(int key);
	Node & getDepot();
	string getInstancePath();
	string getGnuplotPath();
	string getInstanceOut();
	string getInstanceName();
	int getMaxId();
	int getMinId();
	int getNbCustomers();
	int getNbNodes();
	int getNbVehicles();
	int getVehicleCapacity();
	double getAverageCustomerDemand();
	double getTravelledDistance(Node& n1, Node& n2);
	double getTravelledTime(Node& n1, Node& n2);
	/* methods */
	void createBatch();
	void printDistanceMatrix();
	void readData();
	string toString();

private:
	
	void initialiseDistances();
	void initialiseTravelTimes();
	void initialiseConfiguration();
	void initialiseAverageCustomerDemand();
	void initialiseMinMaxId();

	double getDoubleFromLineExactDelimiters(string line, string start, string stop);
	double getDoubleFromLinePartialInitialDelimiter(string line, string start, string stop);

	double calculateTravelledTime(double x1, double y1, double x2, double y2);
	double calculateTravelledDistance(double x1, double y1, double x2, double y2);
	double calculateEuclideanDistance(double x1, double y1, double x2, double y2);

	void orderPoints();
	//void tighteningTW();
};
