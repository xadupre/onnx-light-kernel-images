// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark that applies ParallelFor to an element-wise Abs kernel and
// empirically measures the thread-dispatch overhead that backs
// kParallelForThreadOverheadCost (see onnx_light_kernel_images/parallel_for.h).
//
// Abs is the cheapest possible map kernel (one fabs per element), so it is the
// worst case for the cost model: if parallelising ever pays off for Abs, the
// dispatch overhead has been amortised. The benchmark measures three things:
//
//   1. cost_per_iteration  -- ns per element of the serial Abs kernel;
//   2. Run() dispatch cost -- ns to launch an (almost) empty parallel region,
//                             i.e. the raw wake+join overhead of the workers;
//   3. break-even N        -- the smallest element count at which a forced
//                             parallel Abs actually beats the serial Abs.
//
// From (1) and (3) it back-solves the per-worker overhead the cost model would
// need so that ShouldParallelize() fires exactly at the measured break-even:
//
//     kParallelForThreadOverheadCost ~= break_even_N * cost_per_iter / (P - 1)
//
// Build with -DONNX_LIGHT_KERNEL_IMAGES_BUILD_BENCHMARKS=ON and run the
// resulting bench_parallel_for executable (no arguments).

#include "onnx_light_kernel_images/parallel_for.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using onnx_light_kernel_images::GlobalThreadPool;
using onnx_light_kernel_images::kParallelForThreadOverheadCost;
using onnx_light_kernel_images::ParallelFor;
using onnx_light_kernel_images::ParallelForThreadCount;

using Clock = std::chrono::steady_clock;

double NowNs() {
  return std::chrono::duration<double, std::nano>(Clock::now().time_since_epoch()).count();
}

// Element-wise Abs kernel body over the half-open range [begin, end).
inline void AbsBlock(const float *in, float *out, std::int64_t begin, std::int64_t end) {
  for (std::int64_t i = begin; i < end; ++i) {
    out[i] = std::fabs(in[i]);
  }
}

// Whole-array Abs expressed through the cost-aware ParallelFor. This is the
// intended kernel usage: the cost model decides inline vs. pooled execution.
void AbsParallel(const float *in, float *out, std::int64_t n, double cost_per_iter) {
  ParallelFor(n, cost_per_iter,
              [in, out](std::int64_t begin, std::int64_t end) { AbsBlock(in, out, begin, end); });
}

} // namespace

int main() {
  const std::int64_t threads = ParallelForThreadCount();
  std::printf("hardware threads P = %lld\n", static_cast<long long>(threads));
  std::printf("current kParallelForThreadOverheadCost = %.0f ns\n\n",
              kParallelForThreadOverheadCost);

  // ---- 1. Serial Abs cost per element -------------------------------------
  const std::int64_t n = 1 << 24; // 16M elements, larger than last-level cache.
  std::vector<float> in(static_cast<std::size_t>(n));
  std::vector<float> out(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    in[static_cast<std::size_t>(i)] = (i % 2 ? -1.0f : 1.0f) * static_cast<float>(i);
  }

  double best_serial = 1e30;
  for (int r = 0; r < 5; ++r) {
    const double t0 = NowNs();
    AbsBlock(in.data(), out.data(), 0, n);
    const double t1 = NowNs();
    best_serial = std::min(best_serial, t1 - t0);
  }
  const double cost_per_iter = best_serial / static_cast<double>(n);
  std::printf("Abs cost_per_iteration ~= %.4f ns/elem\n", cost_per_iter);

  // ---- 2. Raw thread-pool dispatch overhead -------------------------------
  if (threads > 1) {
    volatile std::int64_t sink = 0;
    for (int r = 0; r < 50; ++r) {
      GlobalThreadPool().Run(threads, [&sink](std::int64_t b) { sink += b; });
    }
    double best_dispatch = 1e30;
    for (int trial = 0; trial < 10; ++trial) {
      const double t0 = NowNs();
      for (int r = 0; r < 2000; ++r) {
        GlobalThreadPool().Run(threads, [&sink](std::int64_t b) { sink += b; });
      }
      const double t1 = NowNs();
      best_dispatch = std::min(best_dispatch, (t1 - t0) / 2000.0);
    }
    std::printf("thread-pool Run() overhead (%lld blocks) ~= %.0f ns/region"
                " (~%.0f ns/worker)\n",
                static_cast<long long>(threads), best_dispatch,
                best_dispatch / static_cast<double>(threads - 1));
    (void)sink;
  }
  std::printf("\n%-12s %-14s %-14s %-8s\n", "N", "serial_ns", "parallel_ns", "speedup");

  // ---- 3. Empirical break-even for forced-parallel Abs --------------------
  std::int64_t break_even = -1;
  for (std::int64_t size = 256; size <= (1 << 22); size *= 2) {
    double serial = 1e30;
    double parallel = 1e30;
    const std::int64_t block = threads > 0 ? (size + threads - 1) / threads : size;
    for (int r = 0; r < 200; ++r) {
      double t0 = NowNs();
      AbsBlock(in.data(), out.data(), 0, size);
      double t1 = NowNs();
      serial = std::min(serial, t1 - t0);

      t0 = NowNs();
      GlobalThreadPool().Run(threads, [&](std::int64_t b) {
        const std::int64_t begin = b * block;
        if (begin >= size) {
          return;
        }
        AbsBlock(in.data(), out.data(), begin, std::min(begin + block, size));
      });
      t1 = NowNs();
      parallel = std::min(parallel, t1 - t0);
    }
    std::printf("%-12lld %-14.0f %-14.0f %-8.2f\n", static_cast<long long>(size), serial, parallel,
                serial / parallel);
    if (break_even < 0 && parallel < serial) {
      break_even = size;
    }
  }

  std::printf("\nempirical break-even N (parallel beats serial Abs) ~= %lld\n",
              static_cast<long long>(break_even));
  if (break_even > 0 && threads > 1) {
    const double implied =
        static_cast<double>(break_even) * cost_per_iter / static_cast<double>(threads - 1);
    std::printf("implied kParallelForThreadOverheadCost ~= %.0f ns\n", implied);
  }

  // Sanity check that the cost-aware kernel still produces correct output.
  AbsParallel(in.data(), out.data(), n, cost_per_iter);
  bool ok = true;
  for (std::int64_t i = 0; ok && i < n; i += (n / 97 + 1)) {
    ok = out[static_cast<std::size_t>(i)] == std::fabs(in[static_cast<std::size_t>(i)]);
  }
  std::printf("cost-aware AbsParallel output correct: %s\n", ok ? "yes" : "NO");
  return ok ? 0 : 1;
}
