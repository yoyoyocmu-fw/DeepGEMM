"""Actual C++ address/guard checks and abstract skewed-warp synchronization model."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import shutil

SOURCE = Path(__file__).resolve().parents[1] / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh"
ROOT = None
COMPILER = None
WARPS, COPIERS, LOADERS = 24, 8, 4


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cpp_address_checks(source):
    begin = source.index("                    constexpr uint32_t kGateUpTileElements =")
    end = source.index("                    for (uint32_t s = 0;", begin)
    actual = source[begin:end]
    cpp = (
        r"""
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
struct alignas(16) uint4 { uint32_t x[4]; };
using cd_dtype_t = uint16_t;
enum class ActivationType { SiTU, SwiGLU };
struct Copy { const uint4* src; uint4* dst; };
std::vector<Copy> copies;
namespace ptx {
template<int N> void cp_async_cg(const uint4* src, uint4* dst) {
    static_assert(N == 256); copies.push_back({src,dst});
}
}
namespace cutlass { namespace arch { void cp_async_fence() {} } }
uint64_t cases=0, vectors=0, elements=0;
template<int EP, int H, int I, bool BF16, bool Prepared, bool Dispatch, bool SiTU>
void test(bool expected) {
    constexpr uint32_t STORE_BLOCK_M=16, BLOCK_N=128, BLOCK_M=240;
    constexpr uint32_t kHidden=H, kIntermediateHidden=I;
    constexpr uint32_t kNumDgradEpilogueThreads=768;
    constexpr uint32_t SMEM_DISPATCH_SIZE=EP>1 ? 4*H*sizeof(cd_dtype_t) : 0;
    constexpr bool kStageGateUp=BF16, kInputsPrepared=Prepared, kDispatchInputsPrepared=Dispatch;
    constexpr auto kActivationType=SiTU ? ActivationType::SiTU : ActivationType::SwiGLU;
    alignas(1024) static uint8_t smem_buffer[256*1024];
    auto* gemm=smem_buffer+SMEM_DISPATCH_SIZE;
    cd_dtype_t* smem_cd[]={reinterpret_cast<cd_dtype_t*>(gemm),reinterpret_cast<cd_dtype_t*>(gemm+4096)};
    auto* smem_up=reinterpret_cast<cd_dtype_t*>(gemm+8192+5*(120*128+128*64*2));
    std::vector<cd_dtype_t> global(960*2*I);
    const auto* gate_up_output=global.data();
    uint32_t epilogue_thread_idx=0, pool_block_offset=0, m_block_idx=0, n_block_idx=0, valid_m=0;
"""
        + actual
        + r"""
    assert(kPipelineGateUp == expected);
    for (uint32_t valid=0; valid<=240; ++valid) {
      valid_m=valid;
      for (uint32_t pool : {0u,2u}) for (uint32_t n : {0u,1u,23u}) {
        pool_block_offset=pool; n_block_idx=n;
        std::vector<int> seen(valid*128*2,0);
        for (uint32_t s=0; s<(valid+15)/16; ++s) {
          assert(reinterpret_cast<uintptr_t>(gate_tile(s))%16==0);
          assert(reinterpret_cast<uintptr_t>(up_tile(s))%16==0);
          assert(reinterpret_cast<uint8_t*>(gate_tile(s))+4096<=reinterpret_cast<uint8_t*>(up_tile(s)) ||
                 reinterpret_cast<uint8_t*>(up_tile(s))+4096<=reinterpret_cast<uint8_t*>(gate_tile(s)));
          if (expected && (s&1)) {
            assert(reinterpret_cast<uint8_t*>(gate_tile(s))==smem_buffer);
            assert(reinterpret_cast<uint8_t*>(up_tile(s))+4096<=gemm);
          } else {
            assert(gate_tile(s)==smem_cd[1] && up_tile(s)==smem_up);
          }
          copies.clear();
          for (epilogue_thread_idx=0; epilogue_thread_idx<768; ++epilogue_thread_idx) stage_gate_up(s);
          for (auto copy:copies) {
            const auto index=reinterpret_cast<const cd_dtype_t*>(copy.src)-gate_up_output;
            const uint32_t row=index/(2*I)-pool*240, col=index%(2*I);
            const bool is_up=col>=I;
            const uint32_t column=col-(is_up ? I : 0)-n*128;
            assert(row<valid && row/16==s && column<128 && column%8==0);
            auto* expected_dst=(is_up ? up_tile(s) : gate_tile(s))+(row%16)*128+column;
            assert(reinterpret_cast<cd_dtype_t*>(copy.dst)==expected_dst);
            for (uint32_t i=0;i<8;++i) ++seen[(is_up*valid+row)*128+column+i];
            ++vectors; elements+=8;
          }
        }
        for(auto n:seen) assert(n==1);
        ++cases;
      }
    }
}
int main() {
    test<8,3584,3072,true,true,true,true>(true);
    test<1,3584,3072,true,true,true,true>(false);
    test<8,3584,3072,false,true,true,true>(false);
    test<8,3584,3072,true,false,true,true>(false);
    test<8,3584,3072,true,true,false,true>(false);
    test<8,3584,3072,true,true,true,false>(false);
    test<8,4096,3072,true,true,true,true>(false);
    test<8,3584,4096,true,true,true,true>(false);
    std::cout << "{\"passed\":true,\"cases\":" << cases << ",\"vectors\":" << vectors
              << ",\"elements\":" << elements << "}\n";
}
"""
    )
    path = ROOT / "harness/actual-addresses.cpp"
    path.write_text(cpp)
    executable = ROOT / "harness/actual-addresses"
    compiler = COMPILER
    with (ROOT / "reports/compile.log").open("x") as log:
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(path), "-o", str(executable)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads(subprocess.check_output([str(executable)], text=True))


def program(warp, sizes, omit_start=False, omit_join=False, omit_wait=False, omit_next=False, enqueue_last=False):
    result = []
    for tile, valid in enumerate(sizes):
        slices = (valid + 15) // 16
        for s in range(slices):
            tag = (tile, s)
            if not omit_start:
                result.append(("barrier", ("start", *tag)))
            if s == 0 and warp < COPIERS:
                result.append(("issue", tag))
            if warp < LOADERS:
                result.append(("grad_write", tag))
            if warp < COPIERS and not omit_wait:
                result.append(("wait", tag))
            if not omit_join:
                result.append(("barrier", ("join", *tag)))
            if warp < COPIERS and not omit_next and (s + 1 < slices or enqueue_last):
                result.append(("issue", (tile, s + 1)))
            result.append(("read", tag))
    return result


def simulate(sizes, seed, mode="random", **switches):
    rng = random.Random(seed)
    programs = [program(w, sizes, **switches) for w in range(WARPS)]
    pc = [0] * WARPS
    blocked = {}
    arrivals = {}
    pending = {}
    groups = {}
    slots = [[None] * COPIERS for _ in range(2)]
    grad = [None] * LOADERS
    readers = {}
    instructions = 0

    def consumed(tag):
        return tag is None or len(readers.get(tag, set())) == WARPS

    while True:
        ready = []
        for w in range(WARPS):
            if w in blocked or pc[w] >= len(programs[w]):
                continue
            op, tag = programs[w][pc[w]]
            if op == "wait" and (w, tag) in pending:
                continue
            ready.append(w)
        completions = list(pending)
        if not ready and not completions:
            assert all(pc[w] == len(programs[w]) for w in range(WARPS)), "deadlock"
            break
        # Loader-first deliberately defers transfers until every ready warp blocks.
        event = (
            ("warp", min(ready))
            if mode == "loader_first" and ready
            else (
                ("warp", max(ready))
                if mode == "consumer_first" and ready
                else rng.choice([("warp", w) for w in ready] + [("complete", x) for x in completions])
            )
        )
        kind, value = event
        instructions += 1
        if kind == "complete":
            w, tag = value
            slot = tag[1] % 2
            assert consumed(slots[slot][w]), "async overwrite before prior readers finish"
            slots[slot][w] = tag
            del pending[value]
            continue
        w = value
        op, tag = programs[w][pc[w]]
        pc[w] += 1
        if op == "barrier":
            seen = arrivals.setdefault(tag, set())
            assert w not in seen
            seen.add(w)
            blocked[w] = tag
            if len(seen) == WARPS:
                for worker in seen:
                    assert blocked.pop(worker) == tag
                del arrivals[tag]
        elif op == "issue":
            assert (w, tag) not in pending and (w, tag) not in groups, "duplicate copy"
            # No queued copy may still be targeting this part of the same slot.
            assert not any(x[0] == w and x[1][1] % 2 == tag[1] % 2 for x in pending), "pending slot collision"
            pending[(w, tag)] = True
            groups[(w, tag)] = True
        elif op == "wait":
            assert (w, tag) in groups and (w, tag) not in pending, "missing copy or early wait"
            del groups[(w, tag)]
        elif op == "grad_write":
            assert consumed(grad[w]), "TMEM buffer overwritten before prior readers finish"
            grad[w] = tag
        elif op == "read":
            assert all(x == tag for x in slots[tag[1] % 2]), "activation read before copies complete"
            assert all(x == tag for x in grad), "gradient read before TMEM loads complete"
            seen = readers.setdefault(tag, set())
            assert w not in seen
            seen.add(w)
        else:
            raise AssertionError(op)
    assert not groups and not pending, "copy groups not drained at kernel exit"
    assert not blocked and not arrivals
    return instructions


def main():
    global ROOT, COMPILER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for CPU check artifacts")
    parser.add_argument("--cxx", default=shutil.which("c++") or "c++", help="C++17 compiler executable")
    args = parser.parse_args()
    ROOT, COMPILER = args.output, args.cxx
    assert not ROOT.exists(), "Choose a new output directory to preserve earlier evidence"
    source = SOURCE.read_text()
    # Tie the model to the actual caller, not only a proposed schedule.
    section = source[source.index("                    constexpr uint32_t kGateUpTileElements =") :]
    section = section[: section.index("        __syncthreads();")]
    assert section.count("stage_gate_up(0);") == 1 and section.count("stage_gate_up(s + 1);") == 1
    assert section.count("cutlass::arch::NamedBarrier::sync(") == 2
    assert section.index("cutlass::arch::NamedBarrier::sync(") < section.index("stage_gate_up(0);")
    assert section.index("cutlass::arch::cp_async_wait<0>();") < section.index("stage_gate_up(s + 1);")
    assert "if (s + 1 < effective_m / STORE_BLOCK_M)" in section
    assert section.index("stage_gate_up(s + 1);") < section.index("const float gate_unclamped")
    for part in ["harness", "analysis", "reports"]:
        (ROOT / part).mkdir(parents=True, exist_ok=False)
    (ROOT / "harness/check.py").write_bytes(Path(__file__).read_bytes())
    mapping = cpp_address_checks(source)
    cases = [[n] for n in range(241)] + [[0, 1, 17, 240, 16, 239, 0], [240] * 5, [1] * 5, [17] * 5]
    simulations = instructions = 0
    for index, sizes in enumerate(cases):
        for mode in ["random", "loader_first", "consumer_first"]:
            instructions += simulate(sizes, index, mode)
            simulations += 1
    for seed in range(200):
        sizes = [random.Random(seed * 17 + i).randrange(241) for i in range(8)]
        instructions += simulate(sizes, seed)
        simulations += 1
    negative = []
    for name in ["omit_start", "omit_join", "omit_wait", "omit_next", "enqueue_last"]:
        caught = None
        for mode in ["loader_first", "consumer_first", "random"]:
            try:
                simulate([240, 1, 17, 240], 0, mode, **{name: True})
            except AssertionError as error:
                caught = dict(name=name, rejected=True, mode=mode, reason=str(error))
                break
        assert caught, f"Negative control was accepted: {name}"
        negative.append(caught)
    # Isolate final drain from the cross-tile overwrite negative above.
    try:
        simulate([240], 0, "loader_first", enqueue_last=True)
    except AssertionError as error:
        assert str(error) == "copy groups not drained at kernel exit"
        negative.append(dict(name="final_drain", rejected=True, reason=str(error)))
    else:
        raise AssertionError("Final-drain negative control was accepted")
    files = {
        str(p): sha(p)
        for p in [Path(__file__), SOURCE, ROOT / "harness/actual-addresses.cpp", ROOT / "harness/actual-addresses"]
    }
    result = dict(
        passed=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files,
        actual_cpp=mapping,
        simulations=simulations,
        simulated_instructions=instructions,
        negative_controls=negative,
        warp_roles=dict(total=WARPS, copy=COPIERS, tmem_load=LOADERS, activation=WARPS),
        gpu_launched=False,
        limits="Actual extracted C++ guard/copy-address evaluation plus abstract randomized warp/copy/barrier model. Does not prove PTX lowering, hardware ordering/liveness, numerics, latency or runtime memory.",
    )
    (ROOT / "analysis/result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
