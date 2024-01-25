#pragma once

#include "ndzip.hh"

#include <hip/hip_runtime.h>


namespace ndzip {

template<typename T>
class hipified_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~hipified_compressor() = default;

    virtual void compress(const value_type *in_device_data, const extent &data_size, compressed_type *out_device_stream,
            index_type *out_device_stream_length)
            = 0;
};

template<typename T>
class hipified_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~hipified_decompressor() = default;

    virtual void
    decompress(const compressed_type *in_device_stream, value_type *out_device_data, const extent &data_size)
            = 0;
};

template<typename T>
std::unique_ptr<hipified_compressor<T>>
make_hipified_compressor(const compressor_requirements &req, hipStream_t stream = nullptr);

template<typename T>
std::unique_ptr<hipified_decompressor<T>> make_hipified_decompressor(dim_type dims, hipStream_t stream = nullptr);

}  // namespace ndzip
