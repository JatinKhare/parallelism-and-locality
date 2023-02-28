#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;


FILE *fp, *fp2, *fp3;
void save_file(int Z, long int value){
char const *filename = "our_hit_rate_tiling.csv";
fp = fopen(filename,"a+");
fprintf(fp,"%d ", Z);
fprintf(fp,", %ld \n", value);
}
void save_file_2(int Z, long int value){
char const *filename = "our_access_tiling.csv";
fp2 = fopen(filename,"a+");
fprintf(fp2,"%d ", Z);
fprintf(fp2,", %ld \n", value);
}
void save_file_3(int Z, long int value){
char const *filename = "L2_access_tiling.csv";
fp3 = fopen(filename,"a+");
fprintf(fp3,"%d ", Z);
fprintf(fp3,", %ld \n", value);
}
void save_file_4(int Z, float value){
char const *filename = "L2_hits_tiling.csv";
fp3 = fopen(filename,"a+");
fprintf(fp3,"%d ", Z);
fprintf(fp3,", %f \n", value);
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
int final_access;
	// This function returns false if key is not
	// present in cache. Else it moves the key to
	// front by first removing it and then adding
	// it, and returns true.
	bool get(int key) {
		auto it = map.find(key);
		accesses++;
        //cout<<"Value referred: "<< key <<"\n";      
		if (it == map.end()) {
			return false;
		}
        //cout<<"Hit!\n";
		hits++;
        cache.splice(cache.end(), cache, it->second);
		return true;
	}

	void refer(int key, LRUCache* cache2) {
		//cout<<key<<" referenced\n";
        if (get(key)) {
			return;
		}
		else if(cache2->get(key)){
			put(key);
			return;
		}
		else
			cache2->put(key);
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
	int acc_n(){
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
	int M = 8, P = 4, N = 4;
	int Z1 = 21;
	//int Z = 30;
//	cout<<"N = "<<N<<"; Z = "<<Z<<"\n";
 
remove("L2_hits_tiling.csv");


	for(int Z2 = 1; Z2 <=3*M*N*P; Z2++) { //M*P + P*N + M*N; Z++){
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
		LRUCache cache(Z1);
		LRUCache cache2(Z2);
		for(int i=0; i<M; i++) {
	    for(int j=0; j<N; j++) {
		    for(int k=0; k<P; k++) {
			    cache.refer(a[i*P + k], &cache2);
			    cache.refer(b[k*N + j], &cache2);
			    cache.refer(c[i*N + j], &cache2);
		    }
	    }
    }

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
		save_file(Z1, cache.hits_n());
		save_file_2(Z1, cache.acc_n());
		save_file_3(Z2, cache2.acc_n());
		save_file_4(Z2, (float)cache2.hits_n()/(float)cache2.acc_n());
	//	cout<<"Z = <<"<<Z<<"hits = "<<cache.hits_n()<<endl;
	//cout<<"Hit Rate = "<<(float)(3*M*N*P - (M*P +P*N + M*N))/(float)(3*M*P*N)<<endl;
	//cout<<"Hit Rate = "<<(float)(2*M*P*N - M*P - P*N)/(float)(2*M*P*N + N*M)<<endl;
	cout<<"Hit Rate = "<<(float)(M*P*N - N*P)/(float)(M*P + M*P*N + N*M)<<endl;


	}
fclose(fp);
fclose(fp2);
fclose(fp3);
fclose(fp3);

	return 0;
}

// This code is contributed by divyansh2212

