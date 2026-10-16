// Backend engine interface and a reference (CPU) implementation.
#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "infer_engine/batching.h"
#include "infer_engine/sequence.h"

namespace infer_engine {

// Generates one token per active sequence. The C++ reference path is a
// deterministic stand-in; when compiled WITH_CUDA, the concrete engine forwards
// the batch to a TensorRT execution context (INT8/FP16 fused kernels).
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;

  // Execute one iteration; returns sequence id -> generated token ids.
  virtual std::unordered_map<std::string, std::vector<int>> Execute(
      const BatchPlan& plan) = 0;
};

// Deterministic CPU engine used by tests and as a behavioral reference.
class ReferenceEngine : public InferenceEngine {
 public:
  explicit ReferenceEngine(int vocab_size = 32000) : vocab_size_(vocab_size) {}

  std::unordered_map<std::string, std::vector<int>> Execute(
      const BatchPlan& plan) override;

 private:
  int vocab_size_;
};

// Drive a batcher to completion using the given engine. Returns total steps.
int RunToIdle(ContinuousBatcher* batcher, InferenceEngine* engine);

}  // namespace infer_engine
