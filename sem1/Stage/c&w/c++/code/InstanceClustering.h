#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "Common.h"
#include "PbData.h"

using namespace std;

class InstanceClustering
{
	/* attributes */
	Common::user myuser;
	string clusteringName;
	string instancePath;
	string gnuplotPath;
	vector<string> instanceName;
	vector<int> dimension;
	vector<double> dimOverVehicles;
	vector<vector<int>> clusters;
	vector<double> ccDim, ccDimOverVehicle;		/* cluster centroids based on features */

public:

	/* constructors */
	InstanceClustering(Common::user myuser, string clusteringName);

	/* methods */
private:
	void createClusters(int nClusters);
	void initialiseConfiguration();
	void initialiseFeatures();
	void initialiseFeaturesStatic();
	void readInstance();
	void readInstanceStatic();
	void writeClustersGnuPlot();
};
