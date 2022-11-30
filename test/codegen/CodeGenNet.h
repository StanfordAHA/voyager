#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class CodeGenNet : public Network {
 public:
  CodeGenNet(const std::string&);
  ~CodeGenNet(void){};

  std::vector<Workload> getWorkloads(
      const std::vector<std::string>&) const override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> paramsMap;
  std::map<std::string, Files> filesMap;
  std::map<std::string, MemoryMap> memoryMap;

  // Replaces dataDir from Network for more finegrained control
  std::string tensorDir;
  std::string weightDir;
};
