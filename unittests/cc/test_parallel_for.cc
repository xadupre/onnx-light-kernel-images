// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/parallel_for.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

using onnx_light_kernel_images::kParallelForThreadOverheadCost;
using onnx_light_kernel_images::ParallelFor;
using onnx_light_kernel_images::ParallelForThreadCount;
using onnx_light_kernel_images::ShouldParallelize;

namespace {

// Every element of [0, total) is visited exactly once, whether the loop runs
// inline or across the thread pool.
void ExpectFullCoverage(int64_t total, double cost_per_iteration) {
  std::vector<std::atomic<int>> visits(static_cast<size_t>(total));
  for (auto &v : visits) {
    v.store(0, std::memory_order_relaxed);
  }
  ParallelFor(total, cost_per_iteration, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      visits[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed);
    }
  });
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_EQ(visits[static_cast<size_t>(i)].load(std::memory_order_relaxed), 1)
        << "index " << i << " visited the wrong number of times";
  }
}

} // namespace

TEST(ParallelForCostModel, TinyCheapLoopStaysInline) {
  // A handful of near-free iterations must never wake worker threads.
  EXPECT_FALSE(ShouldParallelize(/*total=*/8, /*cost_per_iteration=*/1.0, /*max_threads=*/8));
}

TEST(ParallelForCostModel, ExpensiveLoopParallelizesWhenThreadsAvailable) {
  // Total work far exceeds the dispatch overhead of the extra workers.
  const double cost = kParallelForThreadOverheadCost; // one overhead unit per iteration
  EXPECT_TRUE(ShouldParallelize(/*total=*/1024, cost, /*max_threads=*/8));
}

TEST(ParallelForCostModel, SingleThreadNeverParallelizes) {
  EXPECT_FALSE(ShouldParallelize(/*total=*/1 << 20, /*cost_per_iteration=*/1e6, /*max_threads=*/1));
}

TEST(ParallelForCostModel, NonPositiveInputsNeverParallelize) {
  EXPECT_FALSE(ShouldParallelize(/*total=*/0, /*cost_per_iteration=*/1e9, /*max_threads=*/8));
  EXPECT_FALSE(ShouldParallelize(/*total=*/1000, /*cost_per_iteration=*/0.0, /*max_threads=*/8));
  EXPECT_FALSE(ShouldParallelize(/*total=*/1000, /*cost_per_iteration=*/-1.0, /*max_threads=*/8));
}

TEST(ParallelForCostModel, ThreadCountIsAtLeastOne) { EXPECT_GE(ParallelForThreadCount(), 1); }

TEST(ParallelForExecution, ZeroTotalIsNoOp) {
  bool called = false;
  ParallelFor(0, 1e9, [&](int64_t, int64_t) { called = true; });
  EXPECT_FALSE(called);
}

TEST(ParallelForExecution, CheapLoopCoversRangeInline) {
  ExpectFullCoverage(/*total=*/1000, /*cost_per_iteration=*/0.001);
}

TEST(ParallelForExecution, ExpensiveLoopCoversRange) {
  // High per-iteration cost pushes the cost-aware overload onto the pool when
  // multiple hardware threads are available; the range must still be covered
  // exactly once.
  ExpectFullCoverage(/*total=*/100000, /*cost_per_iteration=*/kParallelForThreadOverheadCost);
}

TEST(ParallelForExecution, ExpensiveLoopComputesCorrectSum) {
  const int64_t total = 200000;
  std::vector<int64_t> data(static_cast<size_t>(total));
  ParallelFor(total, kParallelForThreadOverheadCost, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      data[static_cast<size_t>(i)] = i * 2;
    }
  });
  int64_t expected = 0;
  for (int64_t i = 0; i < total; ++i) {
    expected += i * 2;
  }
  const int64_t got = std::accumulate(data.begin(), data.end(), int64_t{0});
  EXPECT_EQ(got, expected);
}

TEST(ParallelForExecution, GrainSizeOverloadCoversRange) {
  const int64_t total = 100000;
  std::vector<std::atomic<int>> visits(static_cast<size_t>(total));
  for (auto &v : visits) {
    v.store(0, std::memory_order_relaxed);
  }
  ParallelFor(total, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      visits[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed);
    }
  });
  for (int64_t i = 0; i < total; ++i) {
    EXPECT_EQ(visits[static_cast<size_t>(i)].load(std::memory_order_relaxed), 1);
  }
}
