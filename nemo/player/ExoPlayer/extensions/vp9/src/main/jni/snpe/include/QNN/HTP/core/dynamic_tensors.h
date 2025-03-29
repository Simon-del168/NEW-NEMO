//==============================================================================
//
// Copyright (c) 2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DYNAMIC_TENSOR_H
#define DYNAMIC_TENSOR_H
#include "graph_status.h"

#ifdef __cplusplus
struct DynamicStatus {
#endif // __cplusplus
    enum DynamicTensorErrorCode {
        ValidData = 0,
        SemiValidData = 1,
        InvalidData = 2,
        InPlace = 3,
        Fallback = 4,
        NonInplace = 4, // alias to fallback
        InvalidConfig = 5
    };
#ifdef __cplusplus
    static bool skip_execute(const DynamicStatus ec)
    {
        bool retVal;
        switch (DynamicTensorErrorCode(ec)) {
        case ValidData:
        case SemiValidData:
        case Fallback:
            retVal = false;
            break;
        default:
            retVal = true;
            break;
        }
        return retVal;
    }
    DynamicStatus(const DynamicTensorErrorCode ec) : error_code(ec) {}
    explicit DynamicStatus(const int ec) : error_code(static_cast<DynamicTensorErrorCode>(ec)) {}
    bool operator==(const DynamicTensorErrorCode ec) const { return error_code == ec; }
    bool operator!=(const DynamicTensorErrorCode ec) const { return error_code != ec; }
    int to_int() const { return static_cast<int>(error_code); }
    explicit operator DynamicTensorErrorCode() const { return error_code; }
    explicit operator bool() const { return !skip_execute(error_code); }
    static bool failed_execute(const DynamicStatus ec) { return DynamicTensorErrorCode(ec) == InvalidConfig; }

  private:
    DynamicTensorErrorCode error_code;
};

#endif // __cplusplus

template <typename opFuncT, typename valFuncT, opFuncT opFunc, valFuncT valFunc> struct dyn_validation_general;

template <typename... InTypes, typename retType, retType (*opFunc)(InTypes...), DynamicStatus (*valFunc)(InTypes...)>
struct dyn_validation_general<retType (*)(InTypes...), DynamicStatus (*)(InTypes...), opFunc, valFunc> {
    static retType wrapper_impl(InTypes... Inputs)
    {
        auto rc = valFunc(Inputs...);

        if (rc) {
            return opFunc(Inputs...);
        }

        return DynamicStatus::failed_execute(rc) ? GraphStatus::ErrorBadDynamicOp : GraphStatus::Success;
    }
};

template <typename OType, typename IType>
DynamicStatus unary_validation_graph(OType &out, const IType &in, Graph const &graph)
{
    out.set_valid_dims(in.dims().data());
    return out.get_dynamic_state();
}

template <typename OType, typename IType> DynamicStatus unary_validation(OType &out, const IType &in)
{
    out.set_valid_dims(in.dims().data());
    return out.get_dynamic_state();
}

template <typename TensorType> DynamicStatus unary_validation(TensorType &out, const TensorType &in)
{
    out.set_valid_dims(in.dims().data());
    return out.get_dynamic_state();
}

#endif // DYNAMIC_TENSOR_H
