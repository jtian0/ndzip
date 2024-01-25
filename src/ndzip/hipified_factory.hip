#include "hipified_codec.inl"


template<typename T>
std::unique_ptr<ndzip::hipified_compressor<T>>
ndzip::make_hipified_compressor(const compressor_requirements &reqs, hipStream_t stream) {
    return detail::make_with_profile<hipified_compressor, detail::gpu_hipified::hipified_compressor_impl, T>(
            detail::get_dimensionality(reqs), stream, reqs);
}

template<typename T>
std::unique_ptr<ndzip::hipified_decompressor<T>> ndzip::make_hipified_decompressor(dim_type dims, hipStream_t stream) {
    return detail::make_with_profile<hipified_decompressor, detail::gpu_hipified::hipified_decompressor_impl, T>(dims, stream);
}

template<typename T>
std::unique_ptr<ndzip::offloader<T>> ndzip::make_hipified_offloader(dim_type dimensions) {
    return detail::make_with_profile<offloader, detail::gpu_hipified::hipified_offloader, T>(dimensions);
}

namespace ndzip {

template std::unique_ptr<ndzip::hipified_compressor<float>> make_hipified_compressor<float>(
        const compressor_requirements &, hipStream_t);
template std::unique_ptr<ndzip::hip_decompressor<float>> make_hip_decompressor<float>(dim_type, hipStream_t);
template std::unique_ptr<ndzip::hip_compressor<double>> make_hip_compressor<double>(
        const compressor_requirements &, hipStream_t);
template std::unique_ptr<ndzip::hip_decompressor<double>> make_hip_decompressor<double>(dim_type, hipStream_t);
template std::unique_ptr<offloader<float>> make_hip_offloader<float>(dim_type);
template std::unique_ptr<offloader<double>> make_hip_offloader<double>(dim_type);

}  // namespace ndzip
