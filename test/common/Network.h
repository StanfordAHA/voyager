#pragma once

#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

class Network {
 public:
  Network(void){};
  ~Network(void){};

  virtual std::vector<Workload> getWorkloads(
      const std::vector<std::string>&) const = 0;
};