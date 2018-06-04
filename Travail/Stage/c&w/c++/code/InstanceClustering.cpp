#include "InstanceClustering.h"

InstanceClustering::InstanceClustering(Common::user myuser, string clusteringName)
{
	this->myuser = myuser;
	this->clusteringName = clusteringName;
	initialiseConfiguration();
	//readInstanceStatic();
	initialiseFeaturesStatic();
	createClusters(4);
	writeClustersGnuPlot();
}
void InstanceClustering::createClusters(int nClusters)
{
	int nbFeateures = 2;
	int minDim = INT_MAX, maxDim = INT_MIN;
	double minDimOverVehicles = DBL_MAX, maxDimOverVehicles = DBL_MIN;
	double kDist;
		
	/* computation of max and min of features */
	for(int i=0; i<dimension.size(); ++i)
	{
		if(minDim > dimension.at(i)) minDim = dimension.at(i);
		if(maxDim < dimension.at(i)) maxDim = dimension.at(i);
	}

	for(int i=0; i<dimOverVehicles.size(); ++i)
	{
		if(minDimOverVehicles > dimOverVehicles.at(i)) minDimOverVehicles = dimOverVehicles.at(i);
		if(maxDimOverVehicles < dimOverVehicles.at(i)) maxDimOverVehicles = dimOverVehicles.at(i);
	}

	/* computation of first cluster centers */
	for(int i=0; i < (nClusters / nbFeateures); ++i)
	{
		for(int j=0; j<nbFeateures; ++j)
		{
			ccDim.push_back(minDim + (i + 1) * (maxDim - minDim) / ((nClusters / nbFeateures) + 1));
			ccDimOverVehicle.push_back(minDimOverVehicles + (j + 1) * (maxDimOverVehicles - minDimOverVehicles) / (nbFeateures + 1));
		}
	}
	
	do{
		clusters.erase(clusters.begin(), clusters.end());
		/* clusters initialisation */
		vector<int> cl;
		for(int k=0; k<nClusters; ++k) clusters.push_back(cl);

		/* assignment of instances to clusters */
		for(int i=0; i<instanceName.size(); ++i)
		{
			int closerCluster;
			double distMin = DBL_MAX;
			for(int j=0; j<ccDim.size(); ++j)
			{
				double dist = sqrt( pow((dimension.at(i) - ccDim.at(j)), 2 ) + pow((dimOverVehicles.at(i) - ccDimOverVehicle.at(j)), 2 ));
				if(distMin > dist)
				{
					distMin = dist;
					closerCluster = j;
				}
			}
			clusters.at(closerCluster).push_back(i);
		}
		
		writeClustersGnuPlot();

		/* computation of new cluster centroids */
		vector<double> ccNewDim, ccNewDimOverVehicle;
		for(int k=0; k<nClusters; ++k)
		{
			double newDim = 0, newDimOverVehicle = 0;
			for(int i = 0; i<clusters.at(k).size(); ++i)
			{
				newDim += dimension.at(clusters.at(k).at(i));
				newDimOverVehicle += dimOverVehicles.at(clusters.at(k).at(i));
			}
			ccNewDim.push_back(newDim / clusters.at(k).size());
			ccNewDimOverVehicle.push_back(newDimOverVehicle / clusters.at(k).size());
		}

		kDist = 0;
		for(int k=0; k<nClusters; ++k)
		{
			kDist += sqrt( pow (ccDim.at(k) - ccNewDim.at(k), 2) + pow (ccDimOverVehicle.at(k) - ccNewDimOverVehicle.at(k), 2));
		}

		ccDim = ccNewDim;
		ccDimOverVehicle = ccNewDimOverVehicle;

	}while(kDist > 0.1);


}
void InstanceClustering::initialiseConfiguration()
{
	switch (this->myuser)
	{
	case Common::DIEGO: 
		this->instancePath = "C:\\Users\\dcattaru\\Desktop\\Diego\\Instances\\Clustering\\";
		this->gnuplotPath = "C:/Users/dcattaru/Desktop/Diego/Research/on going/011_Tuning/gnuplot/";
		break;
	default:
		break;
	}
}
void InstanceClustering::initialiseFeatures()
{
	for(int i=0; i<instanceName.size(); ++i)
	{
		cout << instanceName.at(i) << endl;
		PbData pbData(this->myuser, instanceName.at(i));
		dimension.push_back(pbData.getNbNodes());
		string vehicle = instanceName.at(i);
		string strToFind = "k";
		vehicle.erase(0, vehicle.find(strToFind) + 1);
		strToFind = ".xml";
		vehicle.erase(vehicle.find(strToFind), strToFind.size());
		int v = stoi(vehicle);
		dimOverVehicles.push_back((double)pbData.getNbNodes()/(double)v);
	}
}
void InstanceClustering::initialiseFeaturesStatic()
{
	instanceName.push_back("A-n32-k05.xml");	dimension.push_back(32); dimOverVehicles.push_back((double)32/5);
	instanceName.push_back("A-n33-k06.xml");	dimension.push_back(33); dimOverVehicles.push_back((double)33/6);
	instanceName.push_back("A-n33-k05.xml");	dimension.push_back(33); dimOverVehicles.push_back((double)33/5);
	instanceName.push_back("A-n34-k05.xml");	dimension.push_back(34); dimOverVehicles.push_back((double)34/5);
	instanceName.push_back("A-n36-k05.xml");	dimension.push_back(36); dimOverVehicles.push_back((double)36/5);
	instanceName.push_back("A-n37-k05.xml");	dimension.push_back(37); dimOverVehicles.push_back((double)37/5);
	instanceName.push_back("A-n37-k06.xml");	dimension.push_back(37); dimOverVehicles.push_back((double)38/6);
	instanceName.push_back("A-n38-k05.xml");	dimension.push_back(38); dimOverVehicles.push_back((double)39/5);
	instanceName.push_back("A-n39-k05.xml");	dimension.push_back(39); dimOverVehicles.push_back((double)39/5);
	instanceName.push_back("A-n39-k06.xml");	dimension.push_back(39); dimOverVehicles.push_back((double)39/6);
	instanceName.push_back("A-n44-k06.xml");	dimension.push_back(44); dimOverVehicles.push_back((double)44/6);
	instanceName.push_back("A-n45-k06.xml");	dimension.push_back(45); dimOverVehicles.push_back((double)45/6);
	instanceName.push_back("A-n45-k07.xml");	dimension.push_back(45); dimOverVehicles.push_back((double)45/7);
	instanceName.push_back("A-n46-k07.xml");	dimension.push_back(46); dimOverVehicles.push_back((double)46/7);
	instanceName.push_back("A-n48-k07.xml");	dimension.push_back(48); dimOverVehicles.push_back((double)48/7);
	instanceName.push_back("A-n53-k07.xml");	dimension.push_back(53); dimOverVehicles.push_back((double)53/7);
	instanceName.push_back("A-n54-k07.xml");	dimension.push_back(54); dimOverVehicles.push_back((double)54/7);
	instanceName.push_back("A-n55-k09.xml");	dimension.push_back(55); dimOverVehicles.push_back((double)55/9);
	instanceName.push_back("A-n60-k09.xml");	dimension.push_back(60); dimOverVehicles.push_back((double)60/9);
	instanceName.push_back("A-n61-k09.xml");	dimension.push_back(61); dimOverVehicles.push_back((double)61/9);
	instanceName.push_back("A-n62-k08.xml");	dimension.push_back(62); dimOverVehicles.push_back((double)62/8);
	instanceName.push_back("A-n63-k09.xml");	dimension.push_back(63); dimOverVehicles.push_back((double)63/9);
	instanceName.push_back("A-n63-k10.xml");	dimension.push_back(63); dimOverVehicles.push_back((double)63/10);
	instanceName.push_back("A-n64-k09.xml");	dimension.push_back(64); dimOverVehicles.push_back((double)64/9);
	instanceName.push_back("A-n65-k09.xml");	dimension.push_back(65); dimOverVehicles.push_back((double)65/9);
	instanceName.push_back("A-n69-k09.xml");	dimension.push_back(69); dimOverVehicles.push_back((double)69/9);
	instanceName.push_back("A-n80-k10.xml");	dimension.push_back(80); dimOverVehicles.push_back((double)80/10);
	/*instanceName.push_back("B-n31-k05.xml");	dimension.push_back(31); dimOverVehicles.push_back((double)31/5);
	instanceName.push_back("B-n34-k05.xml");	dimension.push_back(34); dimOverVehicles.push_back((double)34/5);
	instanceName.push_back("B-n35-k05.xml");	dimension.push_back(35); dimOverVehicles.push_back((double)35/5);
	instanceName.push_back("B-n38-k06.xml");	dimension.push_back(38); dimOverVehicles.push_back((double)38/6);
	instanceName.push_back("B-n39-k05.xml");	dimension.push_back(39); dimOverVehicles.push_back((double)39/5);
	instanceName.push_back("B-n41-k06.xml");	dimension.push_back(41); dimOverVehicles.push_back((double)41/6);
	instanceName.push_back("B-n43-k06.xml");	dimension.push_back(43); dimOverVehicles.push_back((double)43/6);
	instanceName.push_back("B-n44-k07.xml");	dimension.push_back(44); dimOverVehicles.push_back((double)44/7);
	instanceName.push_back("B-n45-k05.xml");	dimension.push_back(45); dimOverVehicles.push_back((double)45/5);
	instanceName.push_back("B-n45-k06.xml");	dimension.push_back(45); dimOverVehicles.push_back((double)45/6);
	instanceName.push_back("B-n50-k07.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/7);
	instanceName.push_back("B-n50-k08.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/8);
	instanceName.push_back("B-n51-k07.xml");	dimension.push_back(51); dimOverVehicles.push_back((double)51/7);
	instanceName.push_back("B-n52-k07.xml");	dimension.push_back(52); dimOverVehicles.push_back((double)52/7);
	instanceName.push_back("B-n56-k07.xml");	dimension.push_back(56); dimOverVehicles.push_back((double)56/7);
	instanceName.push_back("B-n57-k07.xml");	dimension.push_back(57); dimOverVehicles.push_back((double)57/7);
	instanceName.push_back("B-n57-k09.xml");	dimension.push_back(57); dimOverVehicles.push_back((double)57/9);
	instanceName.push_back("B-n63-k10.xml");	dimension.push_back(63); dimOverVehicles.push_back((double)63/10);
	instanceName.push_back("B-n64-k09.xml");	dimension.push_back(64); dimOverVehicles.push_back((double)64/9);
	instanceName.push_back("B-n66-k09.xml");	dimension.push_back(66); dimOverVehicles.push_back((double)66/9);
	instanceName.push_back("B-n67-k10.xml");	dimension.push_back(67); dimOverVehicles.push_back((double)67/10);
	instanceName.push_back("B-n68-k09.xml");	dimension.push_back(68); dimOverVehicles.push_back((double)68/9);
	instanceName.push_back("B-n78-k10.xml");	dimension.push_back(78); dimOverVehicles.push_back((double)78/10);
	instanceName.push_back("E-n022-k04.xml");	dimension.push_back(22); dimOverVehicles.push_back((double)22/4);
	instanceName.push_back("E-n023-k03.xml");	dimension.push_back(23); dimOverVehicles.push_back((double)23/3);
	instanceName.push_back("E-n030-k03.xml");	dimension.push_back(30); dimOverVehicles.push_back((double)30/3);
	instanceName.push_back("E-n033-k04.xml");	dimension.push_back(33); dimOverVehicles.push_back((double)33/4);
	instanceName.push_back("E-n076-k07.xml");	dimension.push_back(76); dimOverVehicles.push_back((double)76/7);
	instanceName.push_back("E-n076-k08.xml");	dimension.push_back(76); dimOverVehicles.push_back((double)76/8);
	instanceName.push_back("E-n076-k14.xml");	dimension.push_back(76); dimOverVehicles.push_back((double)76/14);
	instanceName.push_back("E-n101-k14.xml");	dimension.push_back(101); dimOverVehicles.push_back((double)101/14);
	instanceName.push_back("P-n016-k08.xml");	dimension.push_back(16); dimOverVehicles.push_back((double)16/8);
	instanceName.push_back("P-n019-k02.xml");	dimension.push_back(19); dimOverVehicles.push_back((double)19/2);
	instanceName.push_back("P-n020-k02.xml");	dimension.push_back(20); dimOverVehicles.push_back((double)20/2);
	instanceName.push_back("P-n021-k02.xml");	dimension.push_back(21); dimOverVehicles.push_back((double)21/2);
	instanceName.push_back("P-n022-k02.xml");	dimension.push_back(22); dimOverVehicles.push_back((double)22/2);
	instanceName.push_back("P-n022-k08.xml");	dimension.push_back(22); dimOverVehicles.push_back((double)22/8);
	instanceName.push_back("P-n023-k08.xml");	dimension.push_back(23); dimOverVehicles.push_back((double)23/8);
	instanceName.push_back("P-n040-k05.xml");	dimension.push_back(40); dimOverVehicles.push_back((double)40/5);
	instanceName.push_back("P-n045-k05.xml");	dimension.push_back(45); dimOverVehicles.push_back((double)45/5);
	instanceName.push_back("P-n050-k07.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/7);
	instanceName.push_back("P-n050-k08.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/8);
	instanceName.push_back("P-n050-k10.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/10);
	instanceName.push_back("P-n051-k10.xml");	dimension.push_back(51); dimOverVehicles.push_back((double)51/10);
	instanceName.push_back("P-n055-k07.xml");	dimension.push_back(55); dimOverVehicles.push_back((double)55/7);
	instanceName.push_back("P-n055-k08.xml");	dimension.push_back(55); dimOverVehicles.push_back((double)55/8);
	instanceName.push_back("P-n055-k10.xml");	dimension.push_back(55); dimOverVehicles.push_back((double)55/10);
	instanceName.push_back("P-n055-k15.xml");	dimension.push_back(55); dimOverVehicles.push_back((double)55/15);
	instanceName.push_back("P-n060-k10.xml");	dimension.push_back(60); dimOverVehicles.push_back((double)60/10);
	instanceName.push_back("P-n060-k15.xml");	dimension.push_back(60); dimOverVehicles.push_back((double)60/15);
	instanceName.push_back("P-n070-k10.xml");	dimension.push_back(70); dimOverVehicles.push_back((double)70/10);
	instanceName.push_back("P-n076-k04.xml");	dimension.push_back(76); dimOverVehicles.push_back((double)76/4);
	instanceName.push_back("P-n076-k05.xml");	dimension.push_back(76); dimOverVehicles.push_back((double)76/5);
	instanceName.push_back("CMT01-k05.xml");	dimension.push_back(50); dimOverVehicles.push_back((double)50/5);
	instanceName.push_back("CMT02-k10.xml");	dimension.push_back(75); dimOverVehicles.push_back((double)75/10);
	instanceName.push_back("CMT03-k08.xml");	dimension.push_back(100); dimOverVehicles.push_back((double)100/8);
	instanceName.push_back("CMT04-k12.xml");	dimension.push_back(150); dimOverVehicles.push_back((double)150/12);
	instanceName.push_back("CMT05-k17.xml");	dimension.push_back(199); dimOverVehicles.push_back((double)199/17);
	instanceName.push_back("CMT11-k07.xml");	dimension.push_back(120); dimOverVehicles.push_back((double)120/7);
	instanceName.push_back("CMT12-k10.xml");	dimension.push_back(100); dimOverVehicles.push_back((double)100/10);*/
}
void InstanceClustering::readInstance()
{
	string comment, line, strToFind;
	string index;
	int i;

	ifstream myStream(instancePath + clusteringName);	/* initialization of the stream to the instance address */

	if(!myStream)
	{
		cout << "My dear friend, I encountered an error while opening the file" << endl;
		cout << "Please check again the path to your instances" << endl;
		cout << "Or check if the instance name is well spelled" << endl;
		cout << "Clustering" << endl;
		system("pause");
		exit(1);
	}

	do
	{
		myStream >> comment;
	} 
	while (comment != "<INSTANCES>");
	
	myStream >> comment;
	do
	{
		instanceName.push_back(comment);
		myStream >> comment;
	} 
	while (comment != "</INSTANCES>");

	myStream.close();
	cout << "instance has been red" << endl;
}
void InstanceClustering::readInstanceStatic()
{
	instanceName.push_back("A-n32-k05.xml");
	instanceName.push_back("A-n33-k06.xml");
	instanceName.push_back("A-n33-k05.xml");
	instanceName.push_back("A-n34-k05.xml");
	instanceName.push_back("A-n36-k05.xml");
	instanceName.push_back("A-n37-k05.xml");
	instanceName.push_back("A-n37-k06.xml");
	instanceName.push_back("A-n38-k05.xml");
	instanceName.push_back("A-n39-k05.xml");
	instanceName.push_back("A-n39-k06.xml");
	instanceName.push_back("A-n44-k06.xml");
	instanceName.push_back("A-n45-k06.xml");
	instanceName.push_back("A-n45-k07.xml");
	instanceName.push_back("A-n46-k07.xml");
	instanceName.push_back("A-n48-k07.xml");
	instanceName.push_back("A-n53-k07.xml");
	instanceName.push_back("A-n54-k07.xml");
	instanceName.push_back("A-n55-k09.xml");
	instanceName.push_back("A-n60-k09.xml");
	instanceName.push_back("A-n61-k09.xml");
	instanceName.push_back("A-n62-k08.xml");
	instanceName.push_back("A-n63-k09.xml");
	instanceName.push_back("A-n63-k10.xml");
	instanceName.push_back("A-n64-k09.xml");
	instanceName.push_back("A-n65-k09.xml");
	instanceName.push_back("A-n69-k09.xml");
	instanceName.push_back("A-n80-k10.xml");
	instanceName.push_back("B-n31-k05.xml");
	instanceName.push_back("B-n34-k05.xml");
	instanceName.push_back("B-n35-k05.xml");
	instanceName.push_back("B-n38-k06.xml");
	instanceName.push_back("B-n39-k05.xml");
	instanceName.push_back("B-n41-k06.xml");
	instanceName.push_back("B-n43-k06.xml");
	instanceName.push_back("B-n44-k07.xml");
	instanceName.push_back("B-n45-k05.xml");
	instanceName.push_back("B-n45-k06.xml");
	instanceName.push_back("B-n50-k07.xml");
	instanceName.push_back("B-n50-k08.xml");
	instanceName.push_back("B-n51-k07.xml");
	instanceName.push_back("B-n52-k07.xml");
	instanceName.push_back("B-n56-k07.xml");
	instanceName.push_back("B-n57-k07.xml");
	instanceName.push_back("B-n57-k09.xml");
	instanceName.push_back("B-n63-k10.xml");
	instanceName.push_back("B-n64-k09.xml");
	instanceName.push_back("B-n66-k09.xml");
	instanceName.push_back("B-n67-k10.xml");
	instanceName.push_back("B-n68-k09.xml");
	instanceName.push_back("B-n78-k10.xml");
	instanceName.push_back("E-n022-k04.xml");
	instanceName.push_back("E-n023-k03.xml");
	instanceName.push_back("E-n030-k03.xml");
	instanceName.push_back("E-n033-k04.xml");
	instanceName.push_back("E-n076-k07.xml");
	instanceName.push_back("E-n076-k08.xml");
	instanceName.push_back("E-n076-k14.xml");
	instanceName.push_back("E-n101-k14.xml");
	instanceName.push_back("P-n016-k08.xml");
	instanceName.push_back("P-n019-k02.xml");
	instanceName.push_back("P-n020-k02.xml");
	instanceName.push_back("P-n021-k02.xml");
	instanceName.push_back("P-n022-k02.xml");
	instanceName.push_back("P-n022-k08.xml");
	instanceName.push_back("P-n023-k08.xml");
	instanceName.push_back("P-n040-k05.xml");
	instanceName.push_back("P-n045-k05.xml");
	instanceName.push_back("P-n050-k07.xml");
	instanceName.push_back("P-n050-k08.xml");
	instanceName.push_back("P-n050-k10.xml");
	instanceName.push_back("P-n051-k10.xml");
	instanceName.push_back("P-n055-k07.xml");
	instanceName.push_back("P-n055-k08.xml");
	instanceName.push_back("P-n055-k10.xml");
	instanceName.push_back("P-n055-k15.xml");
	instanceName.push_back("P-n060-k10.xml");
	instanceName.push_back("P-n060-k15.xml");
	instanceName.push_back("P-n070-k10.xml");
	instanceName.push_back("P-n076-k04.xml");
	instanceName.push_back("P-n076-k05.xml");
	instanceName.push_back("CMT01-k05.xml");
	instanceName.push_back("CMT02-k10.xml");
	instanceName.push_back("CMT03-k08.xml");
	instanceName.push_back("CMT04-k12.xml");
	instanceName.push_back("CMT05-k17.xml");
	instanceName.push_back("CMT11-k07.xml");
	instanceName.push_back("CMT12-k10.xml");
}
void InstanceClustering::writeClustersGnuPlot()
{
	ofstream mystream(this->gnuplotPath + "clustering/clustering.dat");
	mystream << "reset" << endl;
	//mystream << "set autoscale" << endl;
	//mystream << "set size square" << endl;
	mystream << "set size ratio -1" << endl;
	mystream << "set output 'C:/Users/dcattaru/Desktop/Diego/Research/on going/011_Tuning/gnuplot/clustering/clust.pdf'" << endl;
	mystream << "set terminal pdf enhanced" << endl;
	mystream << "plot ";
	
	for(int k=0; k<clusters.size(); ++k)
	{
		mystream << "'C:/Users/dcattaru/Desktop/Diego/Research/on going/011_Tuning/gnuplot/clustering/cluster";
		mystream << k;
		mystream << ".dat' pt 1";
		mystream << " notitle ,\\" << endl;
	}
	mystream << "'C:/Users/dcattaru/Desktop/Diego/Research/on going/011_Tuning/gnuplot/clustering/centroids";
	mystream << ".dat' pt 2";
	mystream << " notitle ,\\" << endl;


	mystream.close();

	for(int k=0; k<clusters.size(); ++k)
	{
		string filename ("cluster");
		string extention( ".dat");
		string number(to_string(k));
		filename += number + extention;
		ofstream mystream(this->gnuplotPath + "clustering/" +  filename);
		
		for(int i=0; i<clusters.at(k).size(); ++i)
		{
			mystream << dimension.at(clusters.at(k).at(i)) << "\t";
			mystream << dimOverVehicles.at(clusters.at(k).at(i)) << "\n";
		}
		mystream.close();
	}

	ofstream mystream2(this->gnuplotPath + "clustering/centroids.dat");
	for(int i=0; i<ccDim.size(); ++i)
	{
		mystream2 << ccDim.at(i) << "\t";
		mystream2 << ccDimOverVehicle.at(i) << "\n";
	}
	mystream2.close();

	cout << "instance has been written" << endl;
}