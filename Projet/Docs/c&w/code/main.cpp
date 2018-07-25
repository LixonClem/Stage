#include <iostream>

#include "InstanceClustering.h"
#include "PbData.h"
#include "Solver.h"

using namespace std;

int main(int argc, char** argv)
{

	//InstanceClustering ic(Common::DIEGO, "CVRP.dat");

	string instanceName = "A-n32-k05.xml";
	//string instanceName = "Ediego.xml";
	double lambda = 1, mu = 0, nu = 0;
	if(argc > 1)
	{
		instanceName = argv[1];
	}
	PbData pbData(Common::DIEGO, instanceName);

	Solver solver(&pbData);
	//pbData.createBatch();

	vector<Solution> mysolutions;
	vector<int> allcosts;

	for(int l=0; l<21; ++l)
	{
		for(int m=0; m<21; ++m)
		{
			for(int n=0; n<21; ++n)
			{
				lambda = 0.1 * l;
				mu = 0.1 * m;
				nu = 0.1 * n;
				cout << lambda << "\t" << mu << "\t" << nu << endl;
				Solution solutionBest(&pbData, lambda, mu, nu);
				solver.clarkeAndWrightRnd(solutionBest, lambda, mu, nu);
				solutionBest.control();
				for(int i=0; i<10; ++i) 
				{
					srand(i+1);
					cout << "*******\n";
					Solution solution(&pbData, lambda, mu, nu);
					solver.clarkeAndWrightRnd(solution, lambda, mu, nu);
					solution.control();
					if(solution.getCost() < solutionBest.getCost())
					{
						cout << "best updated\n";
						solutionBest = solution;
					}
					allcosts.push_back(solution.getCost());
				}
				mysolutions.push_back(solutionBest);
			}
		}
	}

	Solution solutionbest(&pbData);
	solutionbest = *mysolutions.begin();
	int pos = 0;

	for(int i=1; i<mysolutions.size(); ++i)
	{
		if(mysolutions.at(i).getCost() < solutionbest.getCost())
		{
			solutionbest = mysolutions.at(i);
			pos = i;
		}
	}

	cout << "best solution cost: " << solutionbest.getCost() << "\t position: " << pos << endl;

/// calcul des distances, à commenter 
	for(int i=0; i<mysolutions.size(); ++i)
	{
		mysolutions.at(i).calculateDistances(solutionbest);
	}
/////

	ofstream mystream(pbData.getInstanceName() + "analysis.dat", ios::app);
//	mystream << "best\t" << solutionbest.getCost() << endl;
	mystream << "l\tm\tn\tcost1\tcost2\tcost3\tcost4\tcost5\tcost6\tcost7\tcost8\tcost9\tcost10"	<< endl;
	mystream.close();
	for(int i=0; i<mysolutions.size(); ++i)
	{
		mysolutions.at(i).write();
			for(int j=(i*10); j<((i+1)*10); ++j)
			{
				mystream <<allcosts.at(j)<< "\t";
			}
		mystream << endl;
	}

	//system("pause");
}
