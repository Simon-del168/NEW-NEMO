//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include "/root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/snpe/include/SNPE/Wrapper.hpp"
#include "/root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/snpe/include/SNPE/DlSystem/String.hpp"

#include "/root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/snpe/include/SNPE/DlContainer/DlContainer.h"
#include "/root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/snpe/include/SNPE/DlSystem/StringList.hpp"



namespace DlContainer {

struct DlcRecord
{
  std::string name;
  std::vector<uint8_t> data;

  DlcRecord()
    : name{},
      data{}
  {  }

  DlcRecord( DlcRecord&& other ) noexcept
    : name(std::move(other.name)),
     data(std::move(other.data))
  {  }
  DlcRecord(const std::string& new_name)
    : name(new_name),
      data()
  {
    if(name.empty()) {
      name.reserve(1);
    }
  }
  DlcRecord(const DlcRecord&) = delete;
};


class IDlContainer :
public Wrapper<IDlContainer, Snpe_DlContainer_Handle_t>
{
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_DlContainer_Delete};

  template<typename StringType>
  void getCatalog_(std::set<StringType>& catalog) const{
    DlSystem::StringList sl(moveHandle(Snpe_DlContainer_GetCatalog(handle())));
    for(auto s : sl){
      catalog.emplace(s);
    }
  }

  class DlcRecordInternal :
  public Wrapper<DlcRecordInternal, Snpe_DlcRecord_Handle_t>
  {
    friend BaseType;
    using BaseType::BaseType;

    static constexpr DeleteFunctionType DeleteFunction{Snpe_DlcRecord_Delete};
  public:
    DlcRecordInternal()
      : BaseType(Snpe_DlcRecord_Create())
    {  }
    explicit DlcRecordInternal(const std::string& name)
    : BaseType(Snpe_DlcRecord_CreateName(name.c_str()))
    {  }

    uint8_t* getData(){
      return Snpe_DlcRecord_Data(handle());
    }
    size_t size() const{
      return Snpe_DlcRecord_Size(handle());
    }
    const char* getName(){
      return Snpe_DlcRecord_Name(handle());
    }
  };


public:
/**
 * @brief Initializes a container from a container archive file.
 *
 * @param[in] filename Container archive file path.
 *
 * @return Status of container open call
 */
  static std::unique_ptr<IDlContainer> open(const std::string& filename) noexcept{
    return makeUnique<IDlContainer>(Snpe_DlContainer_Open(filename.c_str()));
  }

/**
 * @brief Initializes a container from a byte buffer.
 *
 * @param[in] buffer Byte buffer holding the contents of an archive
 *                   file.
 *
 * @param[in] size Size of the byte buffer.
 *
 * @return A Snpe_DlContainer_Handle_t to access the dlContainer
 */
  static std::unique_ptr<IDlContainer> open(const uint8_t* buffer, const size_t size) noexcept{
    return makeUnique<IDlContainer>(Snpe_DlContainer_OpenBuffer(buffer, size));

  }

/**
 * @brief Initializes a container from a byte buffer.
 *
 * @param[in] buffer Byte buffer holding the contents of an archive
 *                   file.
 *
 * @return A Snpe_DlContainer_Handle_t to access the dlContainer
 */
  static std::unique_ptr<IDlContainer> open(const std::vector<uint8_t>& buffer) noexcept{
    return open(buffer.data(), buffer.size());
  }

/**
 * @brief Initializes a container from a container archive file.
 *
 * @param[in] filename Container archive file path.
 *
 * @return Status of container open call
 */
  static std::unique_ptr<IDlContainer> open(const DlSystem::String &filename) noexcept{
    return open(static_cast<const std::string&>(filename));
  }

/**
 * @brief Get the record catalog for a container.
 *
 * @param[in] catalog : Reference to the record catalog if found
 */
  void getCatalog(std::set<std::string>& catalog) const{
    return getCatalog_(catalog);
  }

/**
 * @brief Get the record catalog for a container.
 *
 * @param[in] catalog : Reference to the record catalog if found
 */
  void getCatalog(std::set<DlSystem::String>& catalog) const{
    return getCatalog_(catalog);
  }

/**
 * @brief Get a record from a container by name.
 *
 * @param[in] name : Name of the record to fetch
 * @param[in] record : Reference to the record if found
 *
 * @return A Snpe_DlcRecordHandle_t that owns the record read from the DlContainer
 */
  bool getRecord(const std::string& name, DlcRecord& record) const{
    auto h = Snpe_DlContainer_GetRecord(handle(), name.c_str());
    if(!h) return false;
    DlcRecordInternal internal(moveHandle(h));
    auto data = internal.getData();

    record.name.assign(internal.getName());
    record.data.assign(data, data+internal.size());
    return true;
  }

/**
 * @brief Get a record from a container by name.
 *
 * @param[in] name : Name of the record to fetch
 * @param[in] record : Reference to the record if found
 *
 * @return A Snpe_DlcRecordHandle_t that owns the record read from the DlContainer
 */
  bool getRecord(const DlSystem::String& name, DlcRecord& record) const{
    return getRecord(static_cast<const std::string&>(name), record);
  }

/**
 * @brief Save the container to an archive on disk. This function will save the
 * container if the filename is different from the file that it was opened
 * from, or if at least one record was modified since the container was
 * opened.
 *
 * It will truncate any existing file at the target path.
 *
 * @param[in] filename : Container archive file path.
 *
 * @return indication of success/failure
 */
  bool save(const std::string& filename){
    return Snpe_DlContainer_Save(handle(), filename.c_str());
  }

/**
 * @brief Save the container to an archive on disk. This function will save the
 * container if the filename is different from the file that it was opened
 * from, or if at least one record was modified since the container was
 * opened.
 *
 * It will truncate any existing file at the target path.
 *
 * @param[in] filename : Container archive file path.
 *
 * @return indication of success/failure
 */
  bool save(const DlSystem::String& filename){
    return save(static_cast<const std::string&>(filename));
  }
};

} // ns DlContainer

ALIAS_IN_ZDL_NAMESPACE(DlContainer, DlcRecord)
ALIAS_IN_ZDL_NAMESPACE(DlContainer, IDlContainer)
