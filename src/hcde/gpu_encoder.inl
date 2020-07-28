#pragma once

#include "common.hh"

#ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#   pragma GCC diagnostic ignored "-Wundef"
#endif
#include <SYCL/sycl.hpp>
#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif


namespace hcde::detail {

template<size_t ...Index>
sycl::range<sizeof...(Index)> to_sycl_range(
    const extent<sizeof...(Index)> &e, std::index_sequence<Index...>) {
    return sycl::range<sizeof...(Index)>(e[Index]...);
}

template<unsigned Dims>
sycl::range<Dims> to_sycl_range(const extent<Dims> &e) {
    return extent_to_sycl_range(e, std::make_index_sequence<Dims>{});
}

template<typename Profile>
std::vector<superblock<Profile>> collect_superblocks(const file<Profile> &f) {
    std::vector<superblock<Profile>> sbs;
    sbs.reserve(f.num_superblocks());
    f.for_each_superblock([&](auto sb) { sbs.push_back(sb); });
    return sbs;
}

// SYCL kernel id type
template<typename Profile>
class encode_kernel;

// SYCL kernel id type
template<typename Profile>
class decode_kernel;

}


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::compressed_size_bound(const extent<dimensions> &size) const {
    detail::file<Profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using bits_type = typename Profile::bits_type;
    using hypercube_offset_type = typename Profile::hypercube_offset_type;
    constexpr static auto side_length = Profile::hypercube_side_length;

    sycl::queue q;
    sycl::buffer<data_type> data_buffer(data.data(), sycl::range<1>(data.size().linear_offset()));

    const detail::file<Profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<Profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    std::vector<uint64_t> sb_initial_offset_data;
    uint64_t offset = file.file_header_length();
    for (auto &sb: superblocks) {
        sb_initial_offset_data.push_back(offset);
        offset += file.superblock_header_length() + sb.num_hypercubes() * Profile::compressed_block_size_bound;
    }

    sycl::buffer<uint64_t> sb_initial_offsets(static_cast<const uint64_t*>(sb_initial_offset_data.data()),
        sycl::range<1>{file.num_superblocks()});
    sycl::buffer<std::byte> stream_buffer{sycl::range<1>(compressed_size_bound(data.size()))};
    sycl::buffer<uint64_t> sb_lengths(sycl::range<1>{superblocks.size()});

    q.submit([&](sycl::handler &cgh) {
        auto data_access = data_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto stream_access = stream_buffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto sb_initial_offset_access = sb_initial_offsets.get_access<sycl::access::mode::read>(cgh);
        auto sb_length_access = sb_lengths.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for_work_group<detail::encode_kernel<Profile>>(
            sycl::range<1>(superblocks.size()), sycl::range<1>(file.max_num_hypercubes_per_superblock()),
            [file, superblock_access, data_access, stream_access, sb_length_access, sb_initial_offset_access,
                data_size = data.size()](sycl::group<1> group) {
                const auto sb_id = group.get_id();
                const auto sb = superblock_access[sb_id];

                slice<const data_type, dimensions> device_data(data_access.get_pointer(), data_size);

                std::byte hc_streams[sb.num_hypercubes()][Profile::compressed_block_size_bound + sizeof(bits_type)];
                size_t hc_lengths[sb.num_hypercubes()];
                group.parallel_for_work_item([sb, &device_data, &hc_lengths, &hc_streams](sycl::h_item<1> item) {
                    auto hc_index = item.get_local_id(0);
                    if (hc_index >= sb.num_hypercubes()) {
                        return;
                    }

                    Profile p;
                    bits_type cube[detail::ipow(side_length, Profile::dimensions)];
                    auto offset = sb.hypercube_offset_at(hc_index);
                    auto *cube_pos = cube;
                    detail::for_each_in_hypercube(device_data, offset, side_length,
                        [&](auto &element) {
                            *cube_pos++ = p.load_value(&element);
                        });

                    memset(hc_streams[hc_index], 0, sizeof hc_streams[hc_index]);
                    hc_lengths[hc_index] = p.encode_block(cube, hc_streams[hc_index]);
                });

                hypercube_offset_type hc_offsets[sb.num_hypercubes()];
                auto offset = file.superblock_header_length();
                for (size_t i = 0; i < sb.num_hypercubes(); ++i) {
                    hc_offsets[i] = offset;
                    offset += static_cast<hypercube_offset_type>(hc_lengths[i]);
                }

                sb_length_access[sb_id] = static_cast<uint64_t>(offset);

                std::byte *sb_stream = stream_access.get_pointer() + sb_initial_offset_access[sb_id];
                group.parallel_for_work_item([file, sb, sb_stream, &hc_offsets, &hc_streams, &hc_lengths](
                    sycl::h_item<1> item) {
                    auto hc_index = item.get_local_id(0);
                    if (hc_index < sb.num_hypercubes()) {
                        if (hc_index > 0) {
                            detail::store_unaligned(sb_stream + sizeof(hypercube_offset_type) * (hc_index - 1),
                                detail::endian_transform(hc_offsets[hc_index]));
                        }
                        memcpy(sb_stream + hc_offsets[hc_index], hc_streams[hc_index], hc_lengths[hc_index]);
                    }
                });
            });
    });

    auto file_pos = static_cast<uint64_t>(file.file_header_length());
    auto sb_stream_access = stream_buffer.get_access<sycl::access::mode::read>();
    auto sb_length_access = sb_lengths.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < superblocks.size(); ++i) {
        auto length = sb_length_access[sycl::id<1>{i}];
        if (i > 0) {
            auto file_offset_address = static_cast<char *>(stream) + (i - 1) * sizeof(uint64_t);
            detail::store_unaligned(file_offset_address, detail::endian_transform(file_pos));
        }

        memcpy(static_cast<char *>(stream) + file_pos,
            sb_stream_access.get_pointer() + sb_initial_offset_data[i], length);
        file_pos += length;
    }

    q.wait();

    auto border_offset_address =
        static_cast<char *>(stream) + (file.num_superblocks() - 1) * sizeof(uint64_t);
    detail::store_unaligned(border_offset_address, detail::endian_transform(file_pos));
    file_pos += detail::pack_border(static_cast<char *>(stream) + file_pos, data, side_length);
    return file_pos;
}


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
    const slice<data_type, dimensions> &data) const
{
    using bits_type = typename Profile::bits_type;
    using hypercube_offset_type = typename Profile::hypercube_offset_type;
    constexpr static auto side_length = Profile::hypercube_side_length;

    const detail::file<Profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<Profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    sycl::queue q;
    sycl::buffer<std::byte> stream_buffer(static_cast<const std::byte*>(stream), sycl::range<1>(bytes));
    sycl::buffer<data_type> data_buffer(sycl::range<1>(data.size().linear_offset()));

    q.submit([&](sycl::handler &cgh) {
        auto stream_access = stream_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_access = data_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for_work_group<detail::decode_kernel<Profile>>(sycl::range<1>(superblocks.size()),
            sycl::range<1>(file.max_num_hypercubes_per_superblock()),
                [superblock_access, data_access, stream_access, file, data_size = data.size()](sycl::group<1> group) {
                const auto sb_index = group.get_id(0);
                const auto sb = superblock_access[group.get_id()];

                slice<data_type, dimensions> device_data(data_access.get_pointer(), data_size);
                std::byte *device_stream = stream_access.get_pointer();

                size_t sb_offset = file.file_header_length();
                if (sb_index > 0) {
                    auto sb_offset_address = device_stream + (sb_index - 1) * sizeof(uint64_t);
                    sb_offset = detail::endian_transform(detail::load_unaligned<uint64_t>(sb_offset_address));
                }

                group.parallel_for_work_item([=](sycl::h_item<1> item) {
                    auto hc_index = item.get_local_id(0);
                    if (hc_index >= sb.num_hypercubes()) {
                        return;
                    }

                    size_t hc_offset = file.superblock_header_length();
                    if (hc_index > 0) {
                        auto hc_offset_address = device_stream + sb_offset
                            + (hc_index - 1) * sizeof(hypercube_offset_type);
                        hc_offset = detail::endian_transform(
                            detail::load_unaligned<hypercube_offset_type>(hc_offset_address));
                    }

                    Profile p;
                    bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
                    p.decode_block(device_stream + sb_offset + hc_offset, cube);
                    bits_type *cube_ptr = cube;
                    auto offset = sb.hypercube_offset_at(hc_index);
                    detail::for_each_in_hypercube(device_data, offset, side_length,
                        [&](auto &element) { p.store_value(&element, *cube_ptr++); });
                });
        });
    });

    q.wait();

    auto host_data = data_buffer.template get_access<sycl::access::mode::read>();
    memcpy(data.data(), host_data.get_pointer(), data.size().linear_offset() * sizeof(data_type));

    auto border_pos_address = static_cast<const char *>(stream)
        + (file.num_superblocks() - 1) * sizeof(uint64_t);
    auto border_pos = static_cast<size_t>(detail::endian_transform(
        detail::load_unaligned<uint64_t>(border_pos_address)));
    auto border_length = detail::unpack_border(data, static_cast<const char *>(stream) + border_pos,
        side_length);
    return border_pos + border_length;
}

