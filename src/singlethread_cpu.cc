#include "common.hh"
#include "fast_profile.hh"
#include "strong_profile.hh"


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compressed_size_bound(
        const extent<dimensions> &size) const
{
    detail::file<Profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const
{
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    detail::file<Profile> file(data.size());
    size_t stream_pos = file.file_header_length();

    size_t superblock_index = 0;
    std::vector<bits_type> cube;
    file.for_each_superblock([&](auto superblock) {
        size_t superblock_header_pos = stream_pos;
        stream_pos += file.superblock_header_length();
        size_t superblock_data_pos = stream_pos;

        Profile p;
        cube.resize(detail::ipow(side_length, Profile::dimensions) * superblock.num_hypercubes());
        bits_type *cube_pos = cube.data();
        superblock.for_each_hypercube([&](auto offset) {
            detail::for_each_in_hypercube(data, offset, side_length,
                    [&](auto &element) { *cube_pos++ = p.load_value(&element); });
        });

        size_t hypercube_index = 0;
        cube_pos = cube.data();
        superblock.for_each_hypercube([&](auto) {
            if (hypercube_index > 0) {
                auto offset_address = static_cast<char *>(stream) + superblock_header_pos
                        + (hypercube_index - 1) * sizeof(typename Profile::hypercube_offset_type);
                auto offset = static_cast<typename Profile::hypercube_offset_type>(
                        stream_pos - superblock_data_pos);
                detail::store_unaligned(offset_address, detail::endian_transform(offset));
            }
            stream_pos += p.encode_block(cube_pos, static_cast<char *>(stream) + stream_pos);
            cube_pos += detail::ipow(side_length, Profile::dimensions);
            ++hypercube_index;
        });

        auto file_offset_address = static_cast<char *>(stream)
                + (superblock_index - 1) * sizeof(uint64_t);
        auto file_offset = static_cast<uint64_t>(superblock_header_pos);
        detail::store_unaligned(file_offset_address, detail::endian_transform(file_offset));
        ++superblock_index;
    });

    stream_pos += detail::pack_border(static_cast<char *>(stream) + stream_pos, data, side_length);
    return stream_pos;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
        const slice<data_type, dimensions> &data) const
{
    using bits_type = typename Profile::bits_type;
    constexpr static auto side_length = Profile::hypercube_side_length;
    size_t stream_pos = 0;
    detail::for_each_hypercube_offset(data.size(), side_length, [&](auto offset) {
        Profile p;
        bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
        stream_pos += p.decode_block(static_cast<const char *>(stream) + stream_pos, cube);
        bits_type *cube_ptr = cube;
        detail::for_each_in_hypercube(data, offset, side_length,
                [&](auto &element) { p.store_value(&element, *cube_ptr++); });
    });
    stream_pos += detail::unpack_border(data, static_cast<const char *>(stream) + stream_pos,
            side_length);
    return stream_pos;
}


namespace hcde {

template class singlethread_cpu_encoder<fast_profile<float, 1>>;
template class singlethread_cpu_encoder<fast_profile<float, 2>>;
template class singlethread_cpu_encoder<fast_profile<float, 3>>;
template class singlethread_cpu_encoder<fast_profile<float, 4>>;
template class singlethread_cpu_encoder<strong_profile<float, 1>>;
template class singlethread_cpu_encoder<strong_profile<float, 2>>;
template class singlethread_cpu_encoder<strong_profile<float, 3>>;
template class singlethread_cpu_encoder<strong_profile<float, 4>>;

}
