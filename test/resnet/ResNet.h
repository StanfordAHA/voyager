#pragma once

#include <map>
#include <vector>

#include "test/common/Network.h"
#include "test/common/VerificationTypes.h"

class ResNet : public Network {
 public:
  ResNet() : ResNet("resnet"){};
  ResNet(const std::string &modelName) : ResNet(modelName, "./models/resnet/binary_data/"){};
  ResNet(const std::string &modelName, const std::string &dataDir);
  ~ResNet(void){};

  std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) const override;

  std::vector<Workload> getAllWorkloads() const override;

 private:
  std::vector<std::string> order;
  std::map<std::string, SimplifiedParams> params;
  std::map<std::string, Files> files;

  std::vector<Workload> getWorkloads(const std::vector<std::string> &) const;
};
