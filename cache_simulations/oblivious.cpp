#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;


FILE *fp;
void save_file(int Z, int value){
char const *filename = "our_hit_rate_oblivious.csv";
fp = fopen(filename,"a+");
fprintf(fp,"%d ", Z);
fprintf(fp,", %d \n", value);
}
int Z;

class LRUCache {

private:
	int capacity;
	list<int> cache;
	unordered_map<int, list<int>::iterator> map;
  int hits;
  int accesses;

public:
	LRUCache(int capacity) : capacity(capacity) { hits=0; accesses=0; }

	// This function returns false if key is not
	// present in cache. Else it moves the key to
	// front by first removing it and then adding
	// it, and returns true.
	bool get(int key) {
		auto it = map.find(key);
		accesses++;
        if (it == map.end()) {
			return false;
		}
        //cout<<"Hit!\n";
		hits++;
        cache.splice(cache.end(), cache, it->second);
		return true;
	}

	void refer(int key) {
		//cout<<key<<" referenced\n";
        if (get(key)) {
			return;
		}
		put(key);
	}

	// displays contents of cache in Reverse Order
	void display() {
		for (auto it = cache.rbegin(); it != cache.rend(); ++it) {
		
		// The descendingIterator() method of
		// java.util.LinkedList class is used to return an
		// iterator over the elements in this LinkedList in
		// reverse sequential order
//			cout << *it << " ";
		}
 //     cout<<"\nHits: "<<hits<<"\nAccesses: "<<accesses<<"\n";
	}
	
	float hit_rate(){
		float f_h = hits, f_a = accesses;
//		cout<<"Actual Hit Rate = "<<(f_h/f_a)<<endl;
		return (f_h/f_a);
	}
	int hits_n(){
		return hits;
	}

	void put(int key) {
		if (cache.size() == capacity) {
			int first_key = cache.front();
			cache.pop_front();
			map.erase(first_key);
		}
		cache.push_back(key);
		map[key] = --cache.end();
	}
};

void multiply(int N, int row, int col, vector<int> &a, vector<int> &b, vector<int> &c, LRUCache* cache){
	if(N > 1){
		multiply(N/2, row, col, a, b, c, cache);
		multiply(N/2, row, col+N/2, a, b, c, cache);
		multiply(N/2, row+N/2, col, a, b, c, cache);
		multiply(N/2, row+N/2, col+N/2, a, b, c, cache);
	}
	else{
		for(int i=0; i<N; i++) {
			for(int j=0; j<N; j++) {
				for(int k=0; k<N; k++) {
					//cache.refer(777777+(i*100)+k);
					//cache.refer(2222+k*100+j);
					//cache.refer((i*100)+j);
					cache->refer(a[i*N + k]);
					cache->refer(b[k*N + j]);
					cache->refer(c[i*N + j]);
				}
			}
		}
	}
}

int main() {
	int N = 32;
//	int Z = 10;
//	cout<<"N = "<<N<<"; Z = "<<Z<<"\n";
 

	for(Z = 1; Z <= 3*N*N ; Z++){
    vector<int>a, b, c;
	
    for(int i=0; i<N*N; i++) {
	    a.push_back(i);
    }
    for(int i=0; i<N*N; i++) {
	    b.push_back(i+N*N);
    }
    for(int i=0; i<N*N; i++) {
	    c.push_back(i+2*N*N);
    }
    LRUCache cache(Z);
	
	multiply(N, 0, 0, a, b, c, &cache);
	
    
	cache.display();
	float our_hit_rate;
/*
	if(Z<3){
		cout<<"Our Hit Rate = 0"<<endl;

	}
	else if(Z>=3 && Z <= (2*N + 1)){
		our_hit_rate = (float)(N - 1)/(float)(3*N);
		cout<<" 3 <= Z <= (2N+1) : Our Hit Rate = "<<our_hit_rate<<endl;

	if(cache.hit_rate()==our_hit_rate)
		cout<<"Matched!"<<endl;
	else
		cout<<"[FAILED]"<<endl;
	}

	else if(Z>(2*N + 1) && Z <= (2*N + 1)*N){
		our_hit_rate = (float)(2*(N - 1))/(float)(3*N);
		cout<<"Our Hit Rate = "<<our_hit_rate<<endl;

	if(cache.hit_rate()==our_hit_rate)
		cout<<"Matched!"<<endl;
	else
		cout<<"[FAILED]"<<endl;
	}

	else if(Z>(2*N + 1)*N && Z <= 3*N*N){
		our_hit_rate = (float)(N - 1)/(float)(N);
		cout<<"Our Hit Rate = "<<our_hit_rate<<endl;

	if(cache.hit_rate()==our_hit_rate)
		cout<<"Matched!"<<endl;
	else
		cout<<"[FAILED]"<<endl;
	}
*/
	save_file(Z, cache.hits_n());

//	cout<<"Z = <<"<<Z<<"hits = "<<cache.hits_n()<<endl;
}
fclose(fp);

	return 0;
}

// This code is contributed by divyansh2212

