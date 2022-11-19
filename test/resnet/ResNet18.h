#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class ResNet18 : public Network {
 public:
  ResNet18(void);

  std::vector<Workload> getWorkloads(
      const std::vector<std::string>&) const override;

 private:
  std::map<std::string, SimplifiedParams> paramsMap;
  std::map<std::string, Files> filesMap;
  std::map<std::string, MemoryMap> memoryMap;
  std::vector<std::string> order;
};
