#include <cctype>
#include <cstddef>
#include <cstdio>

#include <wujihandcpp/data/helper.hpp>

#include "wujihandcpp/utility/api.hpp"

namespace wujihandcpp::data {

namespace {

int release_candidate(char pre) {
    return std::toupper(static_cast<unsigned char>(pre)) - 'A';
}

bool is_release_candidate(char pre) {
    const auto upper = std::toupper(static_cast<unsigned char>(pre));
    return 'A' <= upper && upper <= 'Z';
}

} // namespace

WUJIHANDCPP_API size_t FirmwareVersionData::string_length() const {
    int length = 0;
    if (pre == '~') {
        length = std::snprintf(
            nullptr, 0, "%u.%u.%u", unsigned(major), unsigned(minor), unsigned(patch));
    } else if (is_release_candidate(pre)) {
        length = std::snprintf(
            nullptr, 0, "%u.%u.%u-rc%d", unsigned(major), unsigned(minor), unsigned(patch),
            release_candidate(pre));
    } else {
        length = std::snprintf(
            nullptr, 0, "%u.%u.%u-%d", unsigned(major), unsigned(minor), unsigned(patch),
            int(pre));
    }
    return length < 0 ? 0 : static_cast<size_t>(length);
}

WUJIHANDCPP_API void FirmwareVersionData::write_to_string(char* dst) const {
    if (pre == '~') {
        std::sprintf(dst, "%u.%u.%u", unsigned(major), unsigned(minor), unsigned(patch));
    } else if (is_release_candidate(pre)) {
        std::sprintf(
            dst, "%u.%u.%u-rc%d", unsigned(major), unsigned(minor), unsigned(patch),
            release_candidate(pre));
    } else {
        std::sprintf(
            dst, "%u.%u.%u-%d", unsigned(major), unsigned(minor), unsigned(patch), int(pre));
    }
}

} // namespace wujihandcpp::data
