// pybind11 bindings for the native Continuous Batching core.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "infer_engine/batching.h"
#include "infer_engine/engine.h"
#include "infer_engine/sequence.h"

namespace py = pybind11;
using infer_engine::BatchPlan;
using infer_engine::ContinuousBatcher;
using infer_engine::ReferenceEngine;
using infer_engine::RunToIdle;
using infer_engine::Sequence;
using infer_engine::SeqStatus;

PYBIND11_MODULE(_C, m) {
  m.doc() = "Native C++ Continuous Batching core for llm-inference-engine";

  py::enum_<SeqStatus>(m, "SeqStatus")
      .value("WAITING", SeqStatus::kWaiting)
      .value("RUNNING_PREFILL", SeqStatus::kRunningPrefill)
      .value("RUNNING_DECODING", SeqStatus::kRunningDecoding)
      .value("FINISHED_STOP", SeqStatus::kFinishedStop)
      .value("FINISHED_LENGTH", SeqStatus::kFinishedLength);

  py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
      .def(py::init<>())
      .def_readwrite("id", &Sequence::id)
      .def_readwrite("prompt", &Sequence::prompt)
      .def_readonly("output", &Sequence::output)
      .def_readwrite("num_computed", &Sequence::num_computed)
      .def_readwrite("max_new_tokens", &Sequence::max_new_tokens)
      .def_readwrite("priority", &Sequence::priority)
      .def_property_readonly("prompt_len", &Sequence::prompt_len)
      .def_property_readonly("seq_len", &Sequence::seq_len)
      .def_property_readonly("status",
                             [](const Sequence& s) { return static_cast<int>(s.status); });

  py::class_<BatchPlan>(m, "BatchPlan")
      .def_readonly("decode_seqs", &BatchPlan::decode_seqs)
      .def_property_readonly("num_batched_tokens", &BatchPlan::num_batched_tokens);

  py::class_<ContinuousBatcher>(m, "ContinuousBatcher")
      .def(py::init<int, int, int, bool>(), py::arg("max_running") = 256,
           py::arg("prefill_chunk_size") = 2048,
           py::arg("kv_capacity_tokens") = 256 * 2048,
           py::arg("mixed_prefill_decode") = true)
      .def("add",
           [](ContinuousBatcher& self, std::shared_ptr<Sequence> seq) {
             self.AddSequence(std::move(seq));
           })
      .def("running_count", &ContinuousBatcher::running_count)
      .def("waiting_count", &ContinuousBatcher::waiting_count)
      .def("steps", &ContinuousBatcher::steps);

  py::class_<ReferenceEngine>(m, "ReferenceEngine")
      .def(py::init<int>(), py::arg("vocab_size") = 32000);

  m.def("run_to_idle",
        [](ContinuousBatcher& batcher, ReferenceEngine& engine) {
          return RunToIdle(&batcher, &engine);
        },
        "Drive the batcher to completion with the reference engine; returns steps.");
}
