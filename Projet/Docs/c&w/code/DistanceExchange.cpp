#include "DistanceExchange.h"


int myDistanceGreedy (std::vector<unsigned int>& _Sol1, std::vector<unsigned int>& _Sol2){

	int index,indexx;
	int n=0;
	bool zero;
	std::vector<int> Sol1,Sol2;
	std::vector<int> indexStandby;

	for(int k=0;k<_Sol1.size();k++){
		Sol1.push_back(_Sol1[k]);
	}

	for(int k=0;k<_Sol2.size();k++){
		Sol2.push_back(_Sol2[k]);
	}
	
	if(Sol1.size()<Sol2.size()){	
		while(Sol1.size()<Sol2.size())
			Sol1.push_back(0);				
	}

	if(Sol2.size()<Sol1.size()){
		while(Sol2.size()<Sol1.size())		
			Sol2.push_back(0);				
	}
	
	for (int i=1;i<Sol1.size()-1;++i) {	
		if(Sol1[i]!=Sol2[i]) {	
	
			if (indexStandby.size()!=0 && isPresent(Sol2[i],indexStandby,Sol1)) {
				index=getIndexfrom(Sol2[i],indexStandby,Sol1,0);
				swap(Sol1,i,index);
				++n;				
				if(Sol1[index]==Sol2[index]) {				
						indexx=getIndexfrom(index, indexStandby, 0);
						indexStandby.erase(indexStandby.begin()+indexx);
				}
			}
		
			else if (Sol2[i]==0) {
				zero=false;			
				for (int j=i+1;j<Sol1.size()-1;++j) {
					if (Sol1[j]==0 && Sol2[j]==Sol1[i] && zero==false) {					
						swap(Sol1,i,j);
						++n;
						zero=true;	
					}
				}				
				if (zero==false)				
					indexStandby.push_back(i);
			}
		
			else { 
				index=getIndexfrom(Sol2[i], Sol1, i+1); 
				swap(Sol1,i,index);
				++n;		
			}
		
		}		
	}
	
	assert(indexStandby.size()==0);
	assert(Sol1==Sol2);
		
	return n;

}

