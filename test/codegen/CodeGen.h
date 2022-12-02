#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class CodeGen : public Network {
 public:
  CodeGen(const std::string&);
  ~CodeGen(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string>&) const override;

  std::vector<Workload> getAllWorkloads() const override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> paramsMap;
  std::map<std::string, Files> filesMap;
  std::map<std::string, MemoryMap> memoryMap;

  std::vector<Workload> getWorkloads(const std::vector<std::string>&) const;

  // Replaces dataDir from Network for more finegrained control
  std::string tensorDir;
  std::string weightDir;
};
