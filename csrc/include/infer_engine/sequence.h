// Runtime sequence representation for the C++ batching core.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace infer_engine {

enum class SeqStatus {
  kWaiting,
  kRunningPrefill,
  kRunningDecoding,
  kFinishedStop,
  kFinishedLength
};

struct Sequence {
  std::string id;
  std::vector<int> prompt;
  std::vector<int> output;
  SeqStatus status = SeqStatus::kWaiting;
  int num_computed = 0;
  int max_new_tokens = 512;
  int priority = 0;
  int num_preemptions = 0;

  int prompt_len() const { return static_cast<int>(prompt.size()); }
  int seq_len() const { return prompt_len() + static_cast<int>(output.size()); }
  int remaining_prefill() const { return prompt_len() - num_computed; }

  bool is_finished() const {
    return status == SeqStatus::kFinishedStop || status == SeqStatus::kFinishedLength;
  }
  bool is_running() const {
    return status == SeqStatus::kRunningPrefill || status == SeqStatus::kRunningDecoding;
  }
};

}  // namespace infer_engine
