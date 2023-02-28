#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;


FILE *fp, *fp2;
void save_file(int Z, long int value){
char const *filename = "tiling_hits.csv";
fp = fopen(filename,"a+");
fprintf(fp,"%d ", Z);
fprintf(fp,", %ld \n", value);
}
void save_file_2(int Z, long int value){
char const *filename = "tiling_total_accesses.csv";
fp2 = fopen(filename,"a+");
fprintf(fp2,"%d ", Z);
fprintf(fp2,", %ld \n", value);
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
	int final_access = 0;

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

	void refer(int key) {
		final_access++;
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
	int M = 32, P = 8, N = 12, B = 4;
remove("tiling_total_accesses.csv");
remove("tiling_hits.csv");


	for(int Z = 1; Z <= (M*N + N*P + P*N); Z++) { //M*P + P*N + M*N; Z++){
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
		for(int i=0; i<M; i=i+B) {
			for(int j=0; j<N; j=j+B) {
				for(int k=0; k<P; k=k+B) {
					for(int i_b=i; i_b<i+B; i_b++) {
						for(int j_b=j; j_b<j+B; j_b++) {
							for(int k_b=k; k_b<k+B; k_b++) { 
								cache.refer(a[i_b * B * (P/B) + k_b]);
								cache.refer(b[k_b * B * (N/B) + j_b]);
								cache.refer(c[i_b * B * (N/B) + j_b]);
							}
						}
					}
				}
			}
		}
		cache.display();
		save_file(Z, cache.hits_n());
		save_file_2(Z, cache.acc_n());
	}	
fclose(fp);
fclose(fp2);

	return 0;
}

// This code is contributed by divyansh2212

