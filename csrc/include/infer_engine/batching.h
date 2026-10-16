// Continuous Batching scheduler (C++ counterpart of engine/scheduler.py).
#pragma once

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "infer_engine/sequence.h"

namespace infer_engine {

// A single prefill task: the sequence and how many tokens to process this step.
struct PrefillTask {
  Sequence* seq;
  int chunk;
};

// The plan for one engine iteration.
struct BatchPlan {
  std::vector<PrefillTask> prefill_tasks;
  std::vector<Sequence*> decode_seqs;
  std::vector<Sequence*> finished;

  bool empty() const { return prefill_tasks.empty() && decode_seqs.empty(); }
  int num_batched_tokens() const {
    int n = 0;
    for (const auto& t : prefill_tasks) n += t.chunk;
    n += static_cast<int>(decode_seqs.size());
    return n;
  }
};

class ContinuousBatcher {
 public:
  ContinuousBatcher(int max_running = 256,
                    int prefill_chunk_size = 2048,
                    int kv_capacity_tokens = 256 * 2048,
                    bool mixed_prefill_decode = true);

  // Enqueue a new request.
  void AddSequence(std::shared_ptr<Sequence> seq);

  // Compute the next iteration's plan.
  BatchPlan Schedule();

  // Apply the engine's produced tokens (sequence id -> new tokens) for a plan.
  // Returns the sequences that finished during this iteration.
  std::vector<Sequence*> Update(const BatchPlan& plan,
                                const std::unordered_map<std::string, std::vector<int>>& generated);

  bool idle() const { return waiting_.empty() && running_.empty(); }
  int steps() const { return steps_; }
  int running_count() const { return static_cast<int>(running_.size()); }
  int waiting_count() const { return static_cast<int>(waiting_.size()); }

 private:
  std::vector<Sequence*> PreemptFor(int required_tokens);

  std::deque<std::shared_ptr<Sequence>> waiting_;
  std::unordered_map<std::string, std::shared_ptr<Sequence>> running_;

  int max_running_;
  int prefill_chunk_size_;
  int kv_capacity_;
  int used_kv_ = 0;
  bool mixed_;
  int steps_ = 0;
};

}  // namespace infer_engine
