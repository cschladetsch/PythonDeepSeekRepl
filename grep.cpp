#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char **argv) {
    string filename = argv[1];
    ifstream input_file(filename);

    if (!input_file) {
        cerr << "Failed to open the file." << endl;
        return EXIT_FAILURE;
   }

   cout<<"Enter Pattern:" ;
    string searchPattern; cin>>searchPattern;
          while(!input_file.eof())      {
            string readIn; getline (input_file,readIn );        if      (readIn .find(searchPattern)!=std::string::npos){   cout<<readIn    <<endl;}        }return 0; }
