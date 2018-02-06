/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_factory.cc
 * \brief Support for schedule registry.
 */
#include <tvm/tvm.h>
#include <nnvm/base.h>
#include <nnvm/compiler/schedule_factory.h>

#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_set>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(nnvm::compiler::ScheduleFactory);
}  // namespace dmlc

namespace nnvm {
namespace compiler {

tvm::Schedule ScheduleFactory::get_schedule(const tvm::Target& target, const tvm::Array<tvm::Tensor>& outs) {
  for (auto &k : target.keys) {
    auto iter = builders.find(k);
    if (iter != builders.end()) {
      return iter->second(target, outs);
    }
  }
  CHECK(generic_builder != nullptr) << "No generic function registered for " << name;
  return generic_builder(target, outs);
}

}  // namespace compiler
}  // namespace nnvm
