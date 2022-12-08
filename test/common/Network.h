#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

class Network {
 public:
  Network(const std::string &modelName, const std::string &dataDir)
      : modelName(modelName), dataDir(dataDir){};
  virtual ~Network(void) {};

  virtual std::vector<Workload> getWorkloadsInRange(
      const std::vector<std::string> &) const = 0;

  virtual std::vector<Workload> getAllWorkloads() const = 0;

  std::string getDataDir() { return dataDir; }

 protected:
  const std::string &modelName;
  const std::string &dataDir;
};