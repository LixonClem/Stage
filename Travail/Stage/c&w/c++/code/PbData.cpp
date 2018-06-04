#include "PbData.h"
#include "ctype.h"

PbData::PbData(Common::user mMyuser, string mInstanceName)
{
	this->myuser = mMyuser;
	this->instanceName = mInstanceName;
	initialiseConfiguration();
	readData();
	initialiseDistances();
	initialiseTravelTimes();
	initialiseAverageCustomerDemand();
	initialiseMinMaxId();
	//orderPoints();
}
void PbData::createBatch()
{
	string executable = "Tuning.exe\t" + this->getInstanceName() + "\t";
	ofstream mybatch("batch.bat");

	for(int nu = 0; nu < 3; ++nu)
	{
		for(int mu = 0; mu < 3; ++mu)
		{
			for(int lambda = 1; lambda <= 20; lambda++)
			{
				mybatch << executable << 0.1 * lambda << "\t" << mu << "\t" << nu << endl;
			}
		}
	}
	mybatch.close();
}
double PbData::getDoubleFromLineExactDelimiters(string line, string start, string stop)
{
	/* erases exactly start and stop into line */
	line.erase(remove_if(line.begin(), line.end(), isspace, line.end());	/* remove all spaces and tabs */
	line.erase(line.find(start),  start.size());
	line.erase(line.find(stop),  stop.size());
	return stod(line);
}
double PbData::getDoubleFromLinePartialInitialDelimiter(string line, string start, string stop)
{
	/* erases everything there is from 0 and start, then erases exactly stop into line */
	line.erase(remove_if(line.begin(), line.end(), isspace), line.end());	/* remove all spaces and tabs */
	line.erase(0, line.find(start) + start.size());
	line.erase(line.find(stop),  stop.size());
	return stod(line);
}
string PbData::getGnuplotPath()
{
	return this->gnuplotPath;
}
int PbData::getNbCustomers()
{
	return this->nodes.size() - 1;				/* the customers are as many as the points minus the depot */
}
int PbData::getNbNodes()
{
	return this->nodes.size();				/* the customers are as many as the points minus the depot */
}
int PbData::getNbVehicles()
{
	return nbVehicles;
}
int PbData::getVehicleCapacity()
{
	return vehicleCapacity;
}
double PbData::getAverageCustomerDemand()
{
	return this->averageCustomerDemand;
}
Node& PbData::getNode(int key)
{
	return this->nodes.at(key);
}
Node& PbData::getDepot()
{
	return this->depot;
}
string PbData::getInstancePath()
{
	return this->instancePath;
}
string PbData::getInstanceOut()
{
	return this->instanceOut;
}
string PbData::getInstanceName()
{
	return this->instanceName;
}
int PbData::getMaxId()
{
	return this->maxId;
}
int PbData::getMinId()
{
	return this->minId;
}
double PbData::getTravelledDistance(Node& n1, Node& n2)
{
	return this->travelledDistance[n1.getId()][n2.getId()];
}
double PbData::getTravelledTime(Node& n1, Node& n2)
{
	return this->travelledTime[n1.getId()][n2.getId()];
}
void PbData::initialiseAverageCustomerDemand()
{
	this->averageCustomerDemand = 0;
	for(map<int, Node>::iterator it = this->nodes.begin(); it != this->nodes.end(); ++it)
	{
		this->averageCustomerDemand += it->second.getQuantity();
	}
	this->averageCustomerDemand /= (this->nodes.size() - 1);
}
void PbData::initialiseMinMaxId()
{
	map<int, Node>::iterator iter;
	this->maxId = 0;
	this->minId = INT_MAX;
	for(iter = this->nodes.begin(); iter != this->nodes.end(); ++iter)
	{
		if(this->maxId < iter->second.getId())
			this->maxId = iter->second.getId();
		if(this->minId > iter->second.getId())
			this->minId = iter->second.getId();
	}
}
double PbData::calculateTravelledTime(double x1, double y1, double x2, double y2)
{
	return calculateEuclideanDistance(x1, y1, x2, y2);
}
double PbData::calculateTravelledDistance(double x1, double y1, double x2, double y2)
{
	return calculateEuclideanDistance(x1, y1, x2, y2);
}
double PbData::calculateEuclideanDistance(double x1, double y1, double x2, double y2)
{
	return sqrt( pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0) );
}
void PbData::initialiseConfiguration()
{
	switch (this->myuser)
	{
	case Common::DIEGO: 
		this->instancePath = "C:/Users/ismahane/Dropbox/PFE Ismahane/c++/CVRP/";
		this->instanceOut = "";
		this->gnuplotPath = "C:/Users/ismahane/Desktop/gnuplot/";
		break;
	default:
		break;
	}
}
void PbData::initialiseDistances()
{
	map<int, Node>::iterator it1, it2;
	for(it1 = this->nodes.begin(); it1 != this->nodes.end(); ++it1)
	{
		for(it2 = this->nodes.begin(); it2 != this->nodes.end(); ++it2)
		{
			this->travelledDistance[it1->second.getId()][it2->second.getId()] = calculateTravelledDistance(it1->second.getX(), it1->second.getY(), it2->second.getX(), it2->second.getY());
		}
	}
}
void PbData::initialiseTravelTimes()
{
	map<int, Node>::iterator it1, it2;
	for(it1 = this->nodes.begin(); it1 != this->nodes.end(); ++it1)
	{
		for(it2 = this->nodes.begin(); it2 != this->nodes.end(); ++it2)
		{
			double time = sqrt( pow(it1->second.getX() - it2->second.getX(), 2.0) + pow(it1->second.getY() - it2->second.getY(), 2.0) );
			this->travelledTime[it1->second.getId()][it2->second.getId()] = calculateTravelledTime(it1->second.getX(), it1->second.getY(), it2->second.getX(), it2->second.getY());
		}
	}
}
void PbData::orderPoints()
{
	vector<vector<double>> orderDistance;
	for(int i=1; i<=this->getNbNodes(); ++i)
	{
		vector<double> v;
		orderDistance.push_back(v);
		for(int j=1; j<=this->getNbNodes(); ++j)
		{
			this->getTravelledDistance(nodes.at(i), nodes.at(j));
			orderDistance.at(orderDistance.size() - 1).push_back(this->getTravelledDistance(nodes.at(i), nodes.at(j)));
		}
	}

	for(int i=0; i<this->getNbNodes(); ++i)
	{
		sort(orderDistance.at(i).begin(), orderDistance.at(i).end());
	}

	ofstream mystream(this->getGnuplotPath() + "checkClusters/orderDist.dat");
	mystream << "reset" << endl;
	mystream << "set autoscale" <<  endl;
	mystream << "set output 'C:/Users/ismahane/Desktop/gnuplot/checkClusters/clust.pdf'" << endl;
	mystream << "set terminal pdf enhanced" << endl;
	mystream << "plot ";

	for(int i=0; i<this->getNbNodes(); ++i)
	{
		mystream << "'C:/Users/ismahane/Desktop/gnuplot/checkClusters/client" << i << ".dat'";
		mystream << " pt 1 notitle ,\\" << endl;
	} 
	mystream.close();

	for(int i=0; i<this->getNbNodes(); ++i)
	{
		ofstream mystream(this->getGnuplotPath() + "checkClusters/client" + to_string(i) + ".dat");
		for(int j=0; j<this->getNbNodes(); ++j)
		{
			mystream << j << "\t" << to_string(orderDistance.at(i).at(j)) << "\n";
		}
		mystream.close();
	}

}
void PbData::printDistanceMatrix()
{
	for(int i=1; i<=this->getNbNodes(); ++i)
	{
		for(int j=1; j<=this->getNbNodes(); ++j)
		{
			cout << this->getTravelledDistance(this->nodes.at(i), this->nodes.at(j)) << "\t";
		}
		cout << endl;
	}
}
void PbData::readData()
{
	string comment, line, strToFind;
	string index;
	int i;

	ifstream myStream(instancePath + instanceName);	/* initialization of the stream to the instance address */

	if(!myStream)
	{
		cout << "My dear friend, I encountered an error while opening the file" << endl;
		cout << "Please check again the path to your instances" << endl;
		cout << "Or check if the instance name is well spelled" << endl;
		system("pause");
		exit(1);
	}

	strToFind = "<dataset>";
	do
	{
		getline(myStream, line);
	} 
	while (line.find(strToFind) == string::npos);	/* the function string.find finds the first occurence of the text. If it does not find it, it returns npos */
	
	line.erase(line.find(strToFind), strToFind.size());

	strToFind = "</dataset>";
	line.erase(line.find(strToFind, strToFind.size()));
	this->dataSet = line;

	strToFind = "</name>";
	do
	{
		myStream >> comment; 
	} 
	while (comment.find(strToFind) == string::npos);	/* the function string.find finds the first occurence of the text. If it does not find it, it returns npos */
	
	comment.erase(comment.find(strToFind),  strToFind.size());
	strToFind = "<name>";
	comment.erase(comment.find(strToFind), strToFind.size());
	this->instanceName = comment;

	strToFind = "<nodes>";
	do{getline(myStream, line);} 
	while (line.find(strToFind) == string::npos);
	
	strToFind = "</nodes>";
	getline(myStream, line);		/* <node id="i" type="a"> */
	do
	{
		string entry = "<node id=\"";
		line.erase(line.find(entry), entry.size());
		line.erase(remove_if(line.begin(), line.end(), isspace), line.end());	/* remove all spaces and tabs */
		int id = stoi(line.substr(0, line.find("\"")));
		line.erase(0, line.find("\"") + 1);
		entry = "type=\"";
		line.erase(line.find(entry), entry.size());
		int type = stoi(line.substr(0, 1));
		getline(myStream, line);		/* <cx>val</cx> */
		double x = getDoubleFromLineExactDelimiters(line, "<cx>", "</cx>");
		getline(myStream, line);		/* <cy>val</cy> */
		double y = getDoubleFromLineExactDelimiters(line, "<cy>", "</cy>");
		getline(myStream, line);		/* </node> */
		Node n(id, type, x, y);
		if(type == 0) this->depot = n;
		this->nodes.insert(pair<int, Node>(id, n));		/* each node is insert in the map. the key is the id that is unique */
		getline(myStream, line);						/* either </nodes>, then stop, either <node id="i" type="a"> then loop again */
	} 
	while (line.find(strToFind) == string::npos);

	strToFind = "<capacity>";
	do{getline(myStream, line);} 
	while (line.find(strToFind) == string::npos);
	this->vehicleCapacity = (int)getDoubleFromLineExactDelimiters(line, "<capacity>", "</capacity>");

	strToFind = "<requests>";
	do{getline(myStream, line);} 
	while (line.find(strToFind) == string::npos);
	
	strToFind = "</requests>";
	getline(myStream, line);		/* <request id="i" node="id"> */
	do
	{
		int id = (int)getDoubleFromLinePartialInitialDelimiter(line, "node=\"", "\">");
		getline(myStream, line);
		double quantity = getDoubleFromLineExactDelimiters(line, "<quantity>", "</quantity>");
		this->nodes.at(id).setQuantity(quantity);
		this->nodes.at(id).setServiceTime(0);
		getline(myStream, line);		/* </request> */
		getline(myStream, line);		/* either </requests>, then stop, either <request id="i" node="id"> then loop again */
	} 
	while (line.find(strToFind) == string::npos);
	
	myStream.close();
}
string PbData::toString()
{
	string str = "NO INFOS YET!!";
	return str;
}