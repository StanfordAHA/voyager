#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include "universal/number/posit/posit.hpp"

using Real = sw::universal::posit<8, 1>;

int readp(const std::string &filename)
{
  // Read file into vector
  std::ifstream file(filename, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::ostringstream ss;
  ss << file.rdbuf();
  const std::string &s = ss.str();
  std::vector<char> vec(s.begin(), s.end());

assert(vec.size() == 1000);
  int index = 0;
      Real* mx = reinterpret_cast<Real*>(&vec[0]);
  for (int i = 0;i < vec.size();i++)
  {
      Real* posit = reinterpret_cast<Real*>(&vec[i]);

      if (*posit > *mx)
      {
          index = i;
      }
  }

  file.close();

  return index;
}

int main(int argc, char **argv)
{
  // Check the number of parameters
  if (argc != 3)
  {
    // Tell the user how to run the program
    std::cerr << "usage: " << argv[0] << "infile"
              << "outfile" << std::endl;
    return 1;
  }

  std::string infile = std::string(argv[1]);
  std::string outfile = std::string(argv[2]);

  int pos = readp(infile);

  std::ofstream wf(outfile, std::ios::out | std::ios::binary);
  if (!wf.good())
    throw std::runtime_error("File \"" + outfile + "\" does not exist");

std::string postring = std::to_string(pos) + '\n';
  wf.write(postring.c_str(), sizeof(postring.size()));
  wf.close();


    
//   rewrite_data(std::string(argv[1]), std::string(argv[2]));
  return 0;
}