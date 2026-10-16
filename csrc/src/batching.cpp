#include "infer_engine/batching.h"

#include <algorithm>
#include <cassert>

namespace infer_engine {

ContinuousBatcher::ContinuousBatcher(int max_running,
                                     int prefill_chunk_size,
                                     int kv_capacity_tokens,
                                     bool mixed_prefill_decode)
    : max_running_(max_running),
      prefill_chunk_size_(prefill_chunk_size),
      kv_capacity_(kv_capacity_tokens),
      mixed_(mixed_prefill_decode) {}

void ContinuousBatcher::AddSequence(std::shared_ptr<Sequence> seq) {
  seq->status = SeqStatus::kWaiting;
  waiting_.push_back(std::move(seq));
}

BatchPlan ContinuousBatcher::Schedule() {
  ++steps_;
  BatchPlan plan;

  std::vector<Sequence*> decoding;
  std::vector<Sequence*> prefill_in_flight;
  for (auto& kv : running_) {
    if (kv.second->status == SeqStatus::kRunningDecoding) decoding.push_back(kv.second.get());
    if (kv.second->status == SeqStatus::kRunningPrefill) prefill_in_flight.push_back(kv.second.get());
  }

  int decode_tokens = mixed_ ? static_cast<int>(decoding.size()) : 0;
  int budget = std::max(0, prefill_chunk_size_ - decode_tokens);

  // Continue prefill chunks already in flight.
  for (Sequence* seq : prefill_in_flight) {
    if (budget <= 0) break;
    int chunk = std::min(seq->remaining_prefill(), budget);
    plan.prefill_tasks.push_back({seq, chunk});
    budget -= chunk;
  }

  // Admit from the waiting queue.
  while (!waiting_.empty() && budget > 0) {
    if (static_cast<int>(running_.size()) >= max_running_) break;
    auto& seq = waiting_.front();
    int chunk = std::min(seq->remaining_prefill(), budget);
    if (chunk <= 0) break;

    if (used_kv_ + chunk > kv_capacity_) {
      PreemptFor(chunk);
      if (used_kv_ + chunk > kv_capacity_) break;  // could not reclaim enough
    }

    auto owned = std::move(waiting_.front());
    waiting_.pop_front();
    owned->status = SeqStatus::kRunningPrefill;
    Sequence* raw = owned.get();
    running_[owned->id] = std::move(owned);
    used_kv_ += chunk;
    plan.prefill_tasks.push_back({raw, chunk});
    budget -= chunk;
  }

  if (mixed_) {
    plan.decode_seqs = std::move(decoding);
  } else {
    plan.decode_seqs = plan.prefill_tasks.empty() ? std::move(decoding) : std::vector<Sequence*>{};
  }

  return plan;
}

std::vector<Sequence*> ContinuousBatcher::Update(
    const BatchPlan& plan,
    const std::unordered_map<std::string, std::vector<int>>& generated) {

  auto finish = [&](Sequence* s, SeqStatus reason) {
    s->status = reason;
    plan.finished.push_back(s);
  };

  for (const auto& task : plan.prefill_tasks) {
    Sequence* seq = task.seq;
    seq->num_computed += task.chunk;
    auto it = generated.find(seq->id);
    std::vector<int> toks;
    if (it != generated.end()) toks = it->second;
    if (seq->remaining_prefill() == 0) {
      seq->status = SeqStatus::kRunningDecoding;
      for (int t : toks) seq->output.push_back(t);
    }
  }

  for (Sequence* seq : plan.decode_seqs) {
    auto it = generated.find(seq->id);
    if (it != generated.end()) {
      for (int t : it->second) {
        seq->output.push_back(t);
        used_kv_ += 1;
      }
    }
  }

  // Detect completion.
  std::vector<std::string> erase_ids;
  for (auto& kv : running_) {
    Sequence* s = kv.second.get();
    if (s->status != SeqStatus::kRunningDecoding) continue;
    if (static_cast<int>(s->output.size()) >= s->max_new_tokens) {
      finish(s, SeqStatus::kFinishedLength);
      erase_ids.push_back(kv.first);
    } else if (!s->output.empty() && s->output.back() == 0) {
      finish(s, SeqStatus::kFinishedStop);
      erase_ids.push_back(kv.first);
    }
  }

  for (const auto& id : erase_ids) {
    auto it = running_.find(id);
    if (it != running_.end()) {
      used_kv_ -= it->second->num_computed;
      running_.erase(it);
    }
  }

  return plan.finished;
}

std::vector<Sequence*> ContinuousBatcher::PreemptFor(int required_tokens) {
  std::vector<std::pair<std::string, Sequence*>> candidates;
  for (auto& kv : running_) {
    if (kv.second->status == SeqStatus::kRunningDecoding)
      candidates.push_back({kv.first, kv.second.get()});
  }
  // Newest decodable sequences first.
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
              return a.second->priority < b.second->priority ||
                     (a.second->priority == b.second->priority && a.second->num_preemptions < b.second->num_preemptions);
            });

  std::vector<Sequence*> preempted;
  int reclaimed = 0;
  for (auto& c : candidates) {
    if (used_kv_ - reclaimed + required_tokens <= kv_capacity_) break;
    auto node = running_.extract(c.first);
    if (!node) continue;
    std::shared_ptr<Sequence> owned = std::move(node.mapped());
    reclaimed += owned->num_computed;
    owned->num_preemptions += 1;
    owned->status = SeqStatus::kWaiting;
    owned->num_computed = 0;
    owned->output.clear();
    preempted.push_back(owned.get());
    waiting_.push_front(std::move(owned));
  }
  used_kv_ -= reclaimed;
  return preempted;
}

}  // namespace infer_engine
