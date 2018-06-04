#include "DistanceInsert.h"

int myDistanceInsert(std::vector<unsigned int>& _Sol1, std::vector<unsigned int>& _Sol2){

	int length=_Sol1.size()-1;
	std::vector<int> Ctemp(length);
	std::vector<std::vector<int> > C(length,Ctemp);

	for(int k=0; k<length; ++k){
		C[0][k]=0;
		C[k][0]=0;
	}

	for(int i=1;i<length;i++){
		for(int j=1;j<length;j++){
			if(_Sol1[i]==_Sol2[j]){
				C[i][j]=C[i-1][j-1]+1;
			}
			else{
				C[i][j]=std::max(C[i-1][j],C[i][j-1]);
			}
		}
	}

	return length-1-C[length-1][length-1];
}
