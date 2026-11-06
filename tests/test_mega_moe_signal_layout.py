"""CPU canary checks of the actual capacity-sized signal layout.

Only CUDA annotations and unrelated hardware constants are stubbed. Region
addresses and sizes come from the production Workspace, including side-LoRA.
GPU forward/backward and compiler-cache checks remain separate requirements.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestMegaMoESignalLayout(unittest.TestCase):
    def test_capacity_boundaries_and_region_canaries(self) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        header = (ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()
        workspace = header[header.index("namespace deep_gemm::layout {") : header.index("struct Data {")] + "}\n"
        side = (ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe_side_lora.cuh").read_text()
        side = side[side.index("namespace deep_gemm::layout {") :]
        math = (ROOT / "deep_gemm/include/deep_gemm/common/math.cuh").read_text()
        math = (
            math[
                math.index("namespace deep_gemm::math {") : math.index(
                    "template <typename T>\nCUTLASS_DEVICE void swap"
                )
            ]
            + "}\n"
        )
        harness = (
            r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#define DG_UNIFIED_ASSERT(x) assert(x)
#define DG_STATIC_ASSERT(x, text) static_assert(x, text)
namespace deep_gemm::layout {
constexpr uint32_t kNumDeviceLocalityDomains = __NUM_DOMAINS__;
constexpr uint32_t kNumMaxRanks = __MAX_RANKS__;
}
"""
            + math
            + workspace
            + side
            + r"""
using namespace deep_gemm::layout;
struct Case { uint32_t capacity, ring; bool previously_supported; };
int main() {
    for (const auto c : std::vector<Case>{
            {1920, 117120, true}, {132480, 7080960, true},
            {262144, 8388608, true}, {262145, 8388609, false},
            {263040, 14044800, false}}) {
        Workspace sizing(nullptr, 8, 896, c.capacity, 16, c.ring);
        const auto bytes = sizing.get_num_signal_bytes();
        auto* raw = static_cast<unsigned char*>(std::aligned_alloc(128, bytes + 128));
        assert(raw != nullptr);
        std::memset(raw, 0xcd, bytes + 128);
        Workspace w(raw, 8, 896, c.capacity, 16, c.ring);
        SideLoraWorkspace side(raw, 8, 896, c.capacity, 16, c.ring);
        // Test real accessors at both ends of every region, including beyond
        // the former 2^20 ring / 2^15 shared-signal boundaries.
        const uint64_t ring_count = (static_cast<uint64_t>(c.ring) + 7) / 8;
        const uint64_t shared_count = (static_cast<uint64_t>(c.capacity) + 7) / 8;
        struct Region { unsigned char* begin; uint64_t length; };
        std::vector<Region> regions{
            {reinterpret_cast<unsigned char*>(w.get_l1_full_count_ptr()), ring_count*4},
            {reinterpret_cast<unsigned char*>(w.get_l1_empty_count_ptr()), ring_count*4},
            {reinterpret_cast<unsigned char*>(w.get_l2_full_mask_ptr()), ring_count*8},
            {reinterpret_cast<unsigned char*>(w.get_l2_empty_count_ptr()), ring_count*4},
            {reinterpret_cast<unsigned char*>(w.get_shared_l2_full_count_ptr()), shared_count*4}};
        for (size_t i=0; i<regions.size(); ++i) {
            auto r=regions[i];
            assert(reinterpret_cast<uintptr_t>(r.begin)%128==0);
            assert(r.begin >= raw+sizeof(MegaMoESignals));
            assert(r.begin+r.length <= raw+bytes);
            for (size_t j=i+1; j<regions.size(); ++j)
                assert(r.begin+r.length <= regions[j].begin);
            std::memset(r.begin, static_cast<int>(i+1), r.length);
        }
        assert(w.get_l1_full_count_ptr(ring_count-1)[0] == 0x01010101);
        assert(w.get_l1_empty_count_ptr(ring_count-1)[0] == 0x02020202);
        assert(w.get_l2_full_mask_ptr(ring_count-1)[0] == 0x0303030303030303ULL);
        assert(w.get_l2_empty_count_ptr(ring_count-1)[0] == 0x04040404);
        assert(w.get_shared_l2_full_count_ptr(shared_count-1)[0] == 0x05050505);
        *side.get_l2_full_count_ptr(ring_count-1)=0x12345678;
        uint64_t observed_mask;
        std::memcpy(&observed_mask, w.get_l2_full_mask_ptr(ring_count-1), sizeof(observed_mask));
        assert(observed_mask == 0x0303030312345678ULL);
        assert(reinterpret_cast<unsigned char*>(w.get_src_token_topk_idx_ptr()) == raw+bytes);
        *w.get_src_token_topk_idx_ptr()=0xabcdef01;
        for (uint64_t i=0; i<sizeof(MegaMoESignals); ++i) assert(raw[i]==0xcd);
        for (uint64_t i=bytes+4; i<bytes+128; ++i) assert(raw[i]==0xcd);
        const uint64_t old_signal_bytes=sizeof(MegaMoESignals)+(1ULL<<20)*20+(1ULL<<15)*4;
        if (c.previously_supported) assert(bytes<=old_signal_bytes);
        for (const auto block_m : kCandidateBlockM)
            assert((static_cast<uint64_t>(c.ring)+block_m-1)/block_m <= ring_count);
        assert(w.get_num_bytes()>bytes);
        std::cout << c.capacity << " " << c.ring << " " << bytes << "\n";
        std::free(raw);
    }
}
"""
        )
        types = (ROOT / "deep_gemm/include/deep_gemm/common/types.cuh").read_text()
        symmetry = (ROOT / "deep_gemm/include/deep_gemm/layout/sym_buffer.cuh").read_text()
        harness = harness.replace("__NUM_DOMAINS__", re.search(r"kNumDeviceLocalityDomains = (\d+);", types).group(1))
        harness = harness.replace("__MAX_RANKS__", re.search(r"kNumMaxRanks = (\d+);", symmetry).group(1))
        with tempfile.TemporaryDirectory(prefix="megamoe-signal-layout-") as directory:
            binary = str(Path(directory) / "signal-layout")
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-x", "c++", "-", "-o", binary], input=harness, text=True, check=True
            )
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 5)
        self.assertIn("263040 14044800 ", result.stdout)


if __name__ == "__main__":
    unittest.main()
