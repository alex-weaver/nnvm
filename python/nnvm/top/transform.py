# pylint: disable=invalid-name, unused-argument
"""Tensor transformation ops"""
from __future__ import absolute_import

import topi
from .tensor import _fschedule_broadcast, _fschedule_injective
from . import registry as reg
from .registry import OpPattern

# expand_like
@reg.register_compute("expand_like")
def compute_expand_like(attrs, inputs, _):
    """Compute definition of expand_like"""
    exclude = attrs.get_bool("exclude")
    axis = attrs.get_int_tuple("axis")
    if exclude:
        exclude_axis = (axis,) if isinstance(axis, int) else axis
        axis = []
        for item in range(len(inputs[1].shape)):
            if item not in exclude_axis:
                axis.append(item)
        axis = tuple(axis)

    return topi.transform.expand_like(inputs[0], inputs[1], axis)
reg.register_pattern("expand_like", OpPattern.BROADCAST)
reg.register_schedule("expand_like", _fschedule_broadcast)

# reshape_like
@reg.register_compute("reshape_like")
def compute_reshape_like(attrs, inputs, out_info):
    """Compute definition of reshape_like"""
    return topi.reshape(inputs[0], inputs[1].shape)
reg.register_pattern("reshape_like", OpPattern.INJECTIVE)
reg.register_schedule("reshape_like", _fschedule_injective)
