/*!
 *  Copyright (c) 2017 by Contributors
 * \file top/util.h
 * \brief Helpers for registering schedules and ops.
 */
#ifndef NNVM_TOP_UTIL_H_
#define NNVM_TOP_UTIL_H_

#include <tvm/build_module.h>
#include <nnvm/base.h>
#include <nnvm/compiler/op_attr_types.h>

#include <string>

namespace nnvm {
namespace top {

inline nnvm::compiler::FTVMSchedule MakeScheduleQuery(const std::string& name) {
  return [name](const NodeAttrs& attrs,
                const tvm::Array<tvm::Tensor>& outs,
                const std::string& target) {
      auto ctx = tvm::TargetContext(tvm::Target::create(target));
      return tvm::GenericFunc::Get(name)(outs);
  };
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_UTIL_H_
