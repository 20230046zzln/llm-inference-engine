#include "infer_engine/engine.h"

#include <map>

namespace infer_engine {

std::unordered_map<std::string, std::vector<int>> ReferenceEngine::Execute(
    const BatchPlan& plan) {
  std::unordered_map<std::string, std::vector<int>> out;

  // A prefill task that completes a prompt yields the first token; partial chunks
  // return a token too (ignored by the scheduler until prefill completes).
  for (const auto& task : plan.prefill_tasks) {
    const Sequence* s = task.seq;
    int tok = ((s->num_computed + task.chunk) * 7 + 1) % (vocab_size_ - 1);
    tok = tok + 1;  // keep token id != 0 (0 is treated as EOS by the reference)
    out[s->id].push_back(tok);
  }

  for (const Sequence* s : plan.decode_seqs) {
    int tok = ((s->seq_len()) * 13 + 3) % (vocab_size_ - 1);
    out[s->id].push_back(tok + 1);
  }

  return out;
}

int RunToIdle(ContinuousBatcher* batcher, InferenceEngine* engine) {
  while (!batcher->idle()) {
    BatchPlan plan = batcher->Schedule();
    if (plan.empty()) continue;
    auto generated = engine->Execute(plan);
    batcher->Update(plan, generated);
  }
  return batcher->steps();
}

}  // namespace infer_engine
