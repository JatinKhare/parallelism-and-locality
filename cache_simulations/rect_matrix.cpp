#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;


FILE *fp, *fp2;
void save_file(int Z, float value){
char const *filename = "rect_hits.csv";
fp = fopen(filename,"a+");
fprintf(fp,"%d ", Z);
fprintf(fp,", %f \n", value);
}
void save_file_2(int Z, int value){
char const *filename = "rect_accesses.csv";
fp2 = fopen(filename,"a+");
fprintf(fp2,"%d ", Z);
fprintf(fp2,", %d \n", value);
}

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
		return hits/accesses;
	}
	int accesses_n(){
		return accesses;
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

int main() {
	int M = 2, P = 2, N = 2;
	int Z = 0;
	cout<<"N = "<<N<<"; Z = "<<Z<<"\n";
 

	remove("rect_accesses.csv");
	remove("rect_hits.csv");
	for(Z = 1; Z <= 14/*M*P + P*N + M*N*/; Z++){
    vector<int>a, b, c;
	
    for(int i=0; i<M*P; i++) {
	    a.push_back(i);
    }
    for(int i=0; i<P*N; i++) {
	    b.push_back(i+M*P);
    }
    for(int i=0; i<M*N; i++) {
	    c.push_back(i+ M*P + P*N);
    }
    LRUCache cache(Z);
    for(int i=0; i<M; i++) {
	    for(int j=0; j<N; j++) {
		    for(int k=0; k<P; k++) {
			    cache.refer(a[i*P + k]);
			    cache.refer(b[k*N + j]);
			    cache.refer(c[i*N + j]);
		    }
	    }
    }
	cache.display();
	float our_hit_rate;

	if(Z<3){
		our_hit_rate = cache.hit_rate();
	}
	else if(Z>=3 && Z <= (2*P + 1)){
		our_hit_rate = (float)(P - 1)/(float)(3*P);
	}

	else if(Z>(2*P + 1) && Z < (N*P + P + N + 1)){
		our_hit_rate = (float)(2*P*N - P - N)/(float)(3*P*N);
	}

	else if(Z>=(N*P + P + N + 1) && Z <= M*P + N*P + N*M){
		our_hit_rate = (float)(3*M*P*N - (M*P + N*P + N*M))/(float)(3*M*P*N);
	}
	//cout<<"Z = "<<Z<<", Expected = "<<cache.hit_rate()<<", Got = "<<our_hit_rate<<endl;
	assert((cache.hit_rate() - our_hit_rate) < 0.1);
	/*if(cache.hit_rate()==our_hit_rate)
		cout<<"Matched!"<<endl;
	else
		cout<<"[FAILED]"<<endl;*/
	save_file(Z, cache.hit_rate());
	save_file_2(Z, cache.accesses_n());
	fclose(fp);
	fclose(fp2);

	}
	return 0;
}

