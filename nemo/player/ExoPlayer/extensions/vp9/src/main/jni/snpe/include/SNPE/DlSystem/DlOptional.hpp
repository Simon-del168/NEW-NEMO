//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "/root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/snpe/include/SNPE/Wrapper.hpp"

#include <cstddef>
#include <type_traits>


//==============================================================================
//
//  Copyright (c) 2016, 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//#include <cstdio>
#include <utility>
//#include <stdexcept>


namespace DlSystem {


/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * Class to manage a value that may or may not exist. The boolean value
 * of the Optional class is true if the object contains a value and false
 * if it does not contain a value.
 *
 * The class must be evaluated and confirmed as true (containing a value)
 * before being dereferenced.
 */
template <typename T>
class Optional {
public:
  enum class LIFECYCLE {
    NONE = 0,
    REFERENCE_OWNED = 1,
    POINTER_OWNED = 2,
    POINTER_NOT_OWNED = 3
  };

  struct ReferenceCount {
    size_t count = 0;

    void increment() { count++; }

    size_t decrement() {
      if (count > 0) {
        count--;
      }
      return count;
    }
  };

  using U = typename std::remove_pointer<T>::type;

  /**
   * @brief The default constructor is set to not have any value, and is
   * therefore evaluated as false.
   */
  // Do not explicit it so we can return {}
  Optional() {
    m_Type = LIFECYCLE::NONE;
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object v
   * @param[out] Optional instance of object v
   */
  template <typename Q = T>
  Optional (const T& v, typename std::enable_if<!std::is_pointer<Q>::value>::type* = 0)
  : m_Type(LIFECYCLE::REFERENCE_OWNED) {
    try {
      m_StoragePtr = new T(v);
    } catch (...) {
      m_StoragePtr = nullptr;
      m_Type = LIFECYCLE::NONE;
    }
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object v
   * @param[in] type of object
   * @param[in] typename of object
   * @param[out] Optional instance of object v
   */
  template <typename Q = T>
  Optional(U* v, LIFECYCLE type, typename std::enable_if<std::is_pointer<Q>::value>::type* = 0)
  : m_Type(type) {
    switch (m_Type) {
      case LIFECYCLE::POINTER_OWNED:
        m_StoragePtr = v;
        m_Count = new ReferenceCount();
        m_Count->increment();
        break;
      case LIFECYCLE::POINTER_NOT_OWNED:
        m_StoragePtr = v;
        break;
      case LIFECYCLE::REFERENCE_OWNED:
        throw std::bad_exception();
      case LIFECYCLE::NONE:
        break;
    }
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object other
   */
  Optional(const Optional &other) : m_Type(other.m_Type), m_Count(other.m_Count) {
    if (isReference()) {
      m_StoragePtr = new U(*other.m_StoragePtr);
    } else if (isPointer()) {
      m_StoragePtr = other.m_StoragePtr;
      if (isOwned()) {
        m_Count->increment();
      }
    }
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object v
   * @param[out] Optional instance of object v
   */
  Optional& operator=(const Optional& other) noexcept {
    Optional tmp(other);
    swap(std::move(tmp));
    return *this;
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object other
   */
  Optional(Optional&& other) noexcept {
    swap(std::move(other));
  }

  /**
   * @brief Construct an Optional class using an object.
   * @param[in] Reference to an object v
   * @param[out] Optional instance of object v
   */
  Optional& operator=(Optional&& other) noexcept {
    swap(std::move(other));
    return *this;
  }

  /**
   * @brief Destructor of Otional class object.
   */
  ~Optional() {
    if (isOwned()) {
      if (isReference() || (isPointer() && m_Count->decrement() == 0)) {
        delete m_StoragePtr;
        delete m_Count;
      }
    }
  }

  /**
   * @brief Boolean value of Optional class is only true when there exists a value.
   */
  operator bool() const noexcept { return isValid(); }

  /**
   * @brief Boolean value of Optional class is only true when there does not exist a value.
   */
  bool operator!() const noexcept { return !isValid(); }

  /**
   * @brief Get reference of Optional object
   * @warning User must validate Optional has value before.
   */
  const T& operator*() { return this->GetReference(); }

  /**
   * @brief Get reference of Optional object
   * @warning User must validate Optional has value before.
   */
  const T& operator*() const { return this->GetReference(); }

  /**
   * @brief Get reference of Optional object
   * @warning User must validate Optional has value before.
   */
  operator T&() { return this->GetReference(); }

  /**
   * @brief Get reference of Optional object
   * @warning User must validate Optional has value before.
   */
  T operator->() {
    T self = this->GetReference();
    return self;
  }

  /**
   * @brief Releases the ownership of shared pointer m_Count
   */
  void release(){
    if(isOwned() && isPointer()){
      m_Type = LIFECYCLE::POINTER_NOT_OWNED;
      if(m_Count && m_Count->decrement() == 0){
        delete m_Count;
        m_Count = nullptr;
      }
    }
  }
private:
  void swap(Optional&& other) {
    m_Type = other.m_Type;
    m_StoragePtr = other.m_StoragePtr;
    m_Count = other.m_Count;

    other.m_Type = LIFECYCLE::NONE;
    other.m_StoragePtr = nullptr;
    other.m_Count = nullptr;
  }

  template <typename Q = T>
  typename std::enable_if<std::is_same<U, Q>::value, const Q&>::type GetReference() const noexcept {
    if (!isReference()) std::terminate();
    return *static_cast<const Q*>(m_StoragePtr);
  }

  template <typename Q = T>
  typename std::enable_if<std::is_same<U*, Q>::value, const Q&>::type GetReference() const noexcept {
    if (!isPointer()) std::terminate();
    return static_cast<const Q&>(m_StoragePtr);
  }

  template <typename Q = T>
  typename std::enable_if<std::is_same<U, Q>::value, Q&>::type GetReference() noexcept {
    if (!isReference()) std::terminate();
    return *m_StoragePtr;
  }

  template <typename Q = T>
  typename std::enable_if<std::is_same<U*, Q>::value, Q&>::type GetReference() noexcept {
    if (!isPointer()) std::terminate();
    return m_StoragePtr;
  }

  bool isPointer() const {
    return m_Type == LIFECYCLE::POINTER_OWNED || m_Type == LIFECYCLE::POINTER_NOT_OWNED;
  }

  bool isOwned() const {
    return m_Type == LIFECYCLE::REFERENCE_OWNED || m_Type == LIFECYCLE::POINTER_OWNED;
  }

  bool isReference() const {
    return m_Type == LIFECYCLE::REFERENCE_OWNED;
  }

  bool isValid() const {
    return m_Type != LIFECYCLE::NONE;
  }

  U* m_StoragePtr = nullptr;
  LIFECYCLE m_Type;
  ReferenceCount *m_Count = nullptr;
};

} // ns DlSystem

namespace zdl { namespace DlSystem { template<typename T> using Optional = ::DlSystem::Optional<T>; }}
