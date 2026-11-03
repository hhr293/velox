/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <optional>
#include <random>

#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
namespace {

class RollupPartialAggTest : public OperatorTestBase {
 protected:
  static std::string rowText(const RowVectorPtr& rows, vector_size_t row) {
    return rows->toString(row);
  }

  static std::vector<std::string> sortedRows(const RowVectorPtr& rows) {
    std::vector<std::string> result;
    for (vector_size_t i = 0; i < rows->size(); ++i) {
      result.push_back(rowText(rows, i));
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  bool hasOperator(const std::shared_ptr<Task>& task, const std::string& type) {
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == type) {
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(RollupPartialAggTest, fusedSmall) {
  auto input = makeRowVector(
      {"c1", "c2", "x"},
      {
          makeNullableFlatVector<std::string>(
              {"a", "a", "b", std::nullopt, "a"}),
          makeNullableFlatVector<int32_t>({1, 1, 2, 3, std::nullopt}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto makePlan = [&](const std::vector<RowVectorPtr>& data) {
    return PlanBuilder()
        .values(data)
        .expand(
            {{"c1", "c2", "0 as gid", "x"},
             {"c1", "null", "1", "x"},
             {"null", "null", "3", "x"}})
        .partialAggregation({"c1", "c2", "gid"}, {"sum(x)"})
        .planNode();
  };
  for (const auto& data :
       {std::vector<RowVectorPtr>{input},
        std::vector<RowVectorPtr>{
            std::dynamic_pointer_cast<RowVector>(input->slice(0, 0))}}) {
    auto plan = makePlan(data);
    auto original = AssertQueryBuilder(plan).copyResults(pool());
    auto fused =
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kRollupPartialAggregationEnabled, "true")
            .copyResults(pool());
    ASSERT_EQ(original == nullptr, fused == nullptr);
    if (original != nullptr) {
      EXPECT_EQ(sortedRows(original), sortedRows(fused));
    }
  }
}

// Operators of several drivers are created at the same time and every one of
// them has to be fused.
TEST_F(RollupPartialAggTest, fusedMultipleDrivers) {
  auto input = makeRowVector(
      {"c1", "c2", "x"},
      {
          makeNullableFlatVector<std::string>(
              {"a", "a", "b", std::nullopt, "a"}),
          makeNullableFlatVector<int32_t>({1, 1, 2, 3, std::nullopt}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });
  auto plan = PlanBuilder()
                  .values({input}, true)
                  .expand(
                      {{"c1", "c2", "0 as gid", "x"},
                       {"c1", "null", "1", "x"},
                       {"null", "null", "3", "x"}})
                  .partialAggregation({"c1", "c2", "gid"}, {"sum(x)"})
                  .finalAggregation()
                  .planNode();
  auto run = [&](bool fused, bool& hasExpand) {
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .maxDrivers(8)
                      .config(
                          core::QueryConfig::kRollupPartialAggregationEnabled,
                          fused ? "true" : "false")
                      .copyResults(pool(), task);
    hasExpand = hasOperator(task, "Expand");
    return result;
  };
  bool originalHasExpand = false;
  bool fusedHasExpand = true;
  const auto original = run(false, originalHasExpand);
  const auto fused = run(true, fusedHasExpand);
  EXPECT_TRUE(originalHasExpand);
  EXPECT_FALSE(fusedHasExpand);
  EXPECT_EQ(sortedRows(original), sortedRows(fused));
}

// When a flush shows that the tree does not reduce the data, the aggregation is
// abandoned and the rows are passed on as intermediate results. The final
// result does not change.
TEST_F(RollupPartialAggTest, fusedAbandon) {
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < 6; ++batch) {
    constexpr int32_t kSize = 4'000;
    std::vector<std::string> first(kSize);
    std::vector<int32_t> second(kSize);
    std::vector<int64_t> value(kSize);
    for (auto i = 0; i < kSize; ++i) {
      // Every row is a group of its own on all levels but the last one.
      first[i] = fmt::format("key_{}_{}", batch, i);
      second[i] = i % 3;
      value[i] = i;
    }
    batches.push_back(makeRowVector(
        {"c1", "c2", "x"},
        {makeFlatVector<std::string>(first),
         makeFlatVector<int32_t>(second),
         makeFlatVector<int64_t>(value)}));
  }
  // Aggregates that cannot turn a raw row into an intermediate result (the
  // Presto avg) keep the tree.
  auto plan = [&](int variant) {
    return PlanBuilder()
        .values(batches)
        .expand(
            {{"c1", "c2", "0 as gid", "x"},
             {"c1", "null", "1", "x"},
             {"null", "null", "3", "x"}})
        .partialAggregation(
            {"c1", "c2", "gid"},
            variant == 0 ? std::vector<std::string>{"sum(x)"}
                : variant == 1
                ? std::vector<std::string>{"sum(x)", "count(x)", "max(x)"}
                : std::vector<std::string>{"avg(x)"})
        .finalAggregation()
        .planNode();
  };
  for (const int variant : {0, 1, 2}) {
    auto run = [&](bool fused, bool& abandoned) {
      std::shared_ptr<Task> task;
      auto result =
          AssertQueryBuilder(plan(variant))
              .maxDrivers(1)
              .config(
                  core::QueryConfig::kRollupPartialAggregationEnabled,
                  fused ? "true" : "false")
              .config(core::QueryConfig::kMaxPartialAggregationMemory, "100000")
              .config(
                  core::QueryConfig::kMaxExtendedPartialAggregationMemory,
                  "100000")
              .copyResults(pool(), task);
      for (const auto& pipeline : task->taskStats().pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          if (op.runtimeStats.count("rollup.abandoned") != 0) {
            abandoned = true;
          }
        }
      }
      return result;
    };
    bool originalAbandoned = false;
    bool fusedAbandoned = false;
    const auto original = run(false, originalAbandoned);
    const auto fused = run(true, fusedAbandoned);
    EXPECT_FALSE(originalAbandoned);
    EXPECT_EQ(fusedAbandoned, variant != 2) << "variant=" << variant;
    EXPECT_EQ(sortedRows(original), sortedRows(fused)) << "variant=" << variant;
  }
}

// The aggregate argument is computed by a projection between the expansion and
// the aggregation, and the sum is a decimal with a struct intermediate type.
TEST_F(RollupPartialAggTest, fusedDecimalProject) {
  std::mt19937 random(7);
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < 3; ++batch) {
    constexpr int32_t kSize = 3'000;
    std::vector<std::optional<std::string>> first(kSize);
    std::vector<std::optional<int32_t>> second(kSize);
    std::vector<std::optional<std::string>> third(kSize);
    std::vector<std::optional<int32_t>> quantity(kSize);
    std::vector<std::optional<int64_t>> price(kSize);
    for (auto i = 0; i < kSize; ++i) {
      const auto a = random() % 5;
      first[i] = a == 4 ? std::nullopt
                        : std::optional<std::string>(fmt::format("a{}", a));
      second[i] = random() % 7 == 0 ? std::nullopt
                                    : std::optional<int32_t>(random() % 4);
      third[i] = fmt::format("t{}", random() % 40);
      // Groups of the first key 'a0' only get nulls.
      quantity[i] = (a == 0 || random() % 5 == 0)
          ? std::nullopt
          : std::optional<int32_t>(random() % 100);
      price[i] = (a == 0 || random() % 5 == 0)
          ? std::nullopt
          : std::optional<int64_t>(random() % 1'000'000 - 500'000);
    }
    batches.push_back(makeRowVector(
        {"c1", "c2", "c3", "q", "p"},
        {makeNullableFlatVector<std::string>(first),
         makeNullableFlatVector<int32_t>(second),
         makeNullableFlatVector<std::string>(third),
         makeNullableFlatVector<int32_t>(quantity),
         makeNullableFlatVector<int64_t>(price, DECIMAL(7, 2))}));
  }
  auto partialPlan = [&]() {
    return PlanBuilder()
        .values(batches)
        .expand(
            {{"c1", "c2", "c3", "q", "p", "0 as gid"},
             {"c1", "c2", "null", "q", "p", "1"},
             {"c1", "null", "null", "q", "p", "3"},
             {"null", "null", "null", "q", "p", "7"}})
        .project(
            {"c1",
             "c2",
             "c3",
             "gid",
             "coalesce(p * cast(q as decimal(10, 0)), p * cast(1 as decimal(10, 0))) as v"})
        .partialAggregation({"c1", "c2", "c3", "gid"}, {"sum(v)"});
  };
  auto runPlan = [&](const core::PlanNodePtr& plan,
                     bool fused,
                     const std::string& memory) {
    std::shared_ptr<Task> task;
    auto result =
        AssertQueryBuilder(plan)
            .maxDrivers(1)
            .config(
                core::QueryConfig::kRollupPartialAggregationEnabled,
                fused ? "true" : "false")
            .config(core::QueryConfig::kMaxPartialAggregationMemory, memory)
            .config(
                core::QueryConfig::kMaxExtendedPartialAggregationMemory, memory)
            .copyResults(pool(), task);
    EXPECT_EQ(hasOperator(task, "Expand"), !fused);
    EXPECT_EQ(hasOperator(task, "FilterProject"), !fused);
    return result;
  };

  // Partial results are identical, including the empty flag of the sum state.
  const auto partial = partialPlan().planNode();
  const auto original = runPlan(partial, false, "17179869184");
  const auto fused = runPlan(partial, true, "17179869184");
  EXPECT_GT(original->size(), 0);
  EXPECT_EQ(sortedRows(original), sortedRows(fused));

  // With a small limit the partial results differ, the final ones do not.
  const auto final = partialPlan().finalAggregation().planNode();
  EXPECT_EQ(
      sortedRows(runPlan(final, false, "17179869184")),
      sortedRows(runPlan(final, true, "20000")));
}

// Several aggregates of different kinds, some over computed arguments and some
// over input columns, give the same final result as the unfused plan, also with
// flushes and with an empty input.
TEST_F(RollupPartialAggTest, fusedMultiAggregate) {
  std::mt19937 random(11);
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < 3; ++batch) {
    constexpr int32_t kSize = 3'000;
    std::vector<std::optional<std::string>> first(kSize);
    std::vector<std::optional<int32_t>> second(kSize);
    std::vector<std::optional<std::string>> third(kSize);
    std::vector<std::optional<int64_t>> bigint(kSize);
    std::vector<std::optional<int32_t>> integer(kSize);
    std::vector<std::optional<int64_t>> decimal(kSize);
    for (auto i = 0; i < kSize; ++i) {
      const auto a = random() % 5;
      first[i] = a == 4 ? std::nullopt
                        : std::optional<std::string>(fmt::format("a{}", a));
      second[i] = random() % 7 == 0 ? std::nullopt
                                    : std::optional<int32_t>(random() % 4);
      third[i] = fmt::format("t{}", random() % 40);
      // Groups of the first key 'a0' only get nulls.
      const bool allNull = a == 0;
      bigint[i] = (allNull || random() % 5 == 0)
          ? std::nullopt
          : std::optional<int64_t>(random() % 1'000'000 - 500'000);
      integer[i] = (allNull || random() % 5 == 0)
          ? std::nullopt
          : std::optional<int32_t>(random() % 100);
      decimal[i] = (allNull || random() % 5 == 0)
          ? std::nullopt
          : std::optional<int64_t>(random() % 1'000'000 - 500'000);
    }
    batches.push_back(makeRowVector(
        {"c1", "c2", "c3", "b", "q", "p"},
        {makeNullableFlatVector<std::string>(first),
         makeNullableFlatVector<int32_t>(second),
         makeNullableFlatVector<std::string>(third),
         makeNullableFlatVector<int64_t>(bigint),
         makeNullableFlatVector<int32_t>(integer),
         makeNullableFlatVector<int64_t>(decimal, DECIMAL(12, 2))}));
  }
  const std::vector<std::vector<std::string>> sets = {
      {"c1", "c2", "c3", "b", "q", "p", "0 as gid"},
      {"c1", "c2", "null", "b", "q", "p", "1"},
      {"c1", "null", "null", "b", "q", "p", "3"},
      {"null", "null", "null", "b", "q", "p", "7"}};
  const std::vector<std::string> keys = {"c1", "c2", "c3", "gid"};
  const std::vector<std::string> aggregates = {
      "sum(b)", "min(q)", "max(p)", "avg(q)", "avg(p)", "count(q)", "sum(v)"};
  auto makePlan = [&](const std::vector<RowVectorPtr>& data,
                      bool withProject,
                      bool final) {
    auto builder = PlanBuilder().values(data).expand(sets);
    if (withProject) {
      builder.project(
          {"c1",
           "c2",
           "c3",
           "gid",
           "b",
           "q",
           "p",
           "coalesce(p * cast(q as decimal(10, 0)), p * cast(1 as decimal(10, 0))) as v"});
    }
    auto aggregateList = aggregates;
    if (!withProject) {
      // Without a projection every argument is an input column.
      aggregateList.back() = "sum(b)";
    }
    builder.partialAggregation(keys, aggregateList);
    if (final) {
      builder.finalAggregation();
    }
    return builder.planNode();
  };
  int64_t numFlushes = 0;
  auto runPlan = [&](const core::PlanNodePtr& plan,
                     bool fused,
                     const std::string& memory) {
    std::shared_ptr<Task> task;
    auto result =
        AssertQueryBuilder(plan)
            .maxDrivers(1)
            .config(
                core::QueryConfig::kRollupPartialAggregationEnabled,
                fused ? "true" : "false")
            .config(core::QueryConfig::kMaxPartialAggregationMemory, memory)
            .config(
                core::QueryConfig::kMaxExtendedPartialAggregationMemory, memory)
            .copyResults(pool(), task);
    EXPECT_EQ(hasOperator(task, "Expand"), !fused);
    numFlushes = 0;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (const auto it = op.runtimeStats.find("rollup.numFlushes");
            it != op.runtimeStats.end()) {
          numFlushes = it->second.sum;
        }
      }
    }
    return result;
  };

  const auto empty = std::vector<RowVectorPtr>{
      std::dynamic_pointer_cast<RowVector>(batches[0]->slice(0, 0))};
  for (const bool withProject : {true, false}) {
    // The partial states are identical when nothing is flushed.
    const auto partial = makePlan(batches, withProject, false);
    const auto original = runPlan(partial, false, "17179869184");
    EXPECT_GT(original->size(), 0);
    EXPECT_EQ(
        sortedRows(original),
        sortedRows(runPlan(partial, true, "17179869184")));

    // With a small limit the partial results differ, the final ones do not.
    const auto final = makePlan(batches, withProject, true);
    const auto expected = sortedRows(runPlan(final, false, "17179869184"));
    EXPECT_EQ(expected, sortedRows(runPlan(final, true, "17179869184")));
    EXPECT_EQ(expected, sortedRows(runPlan(final, true, "20000")));
    EXPECT_GT(numFlushes, 1);

    const auto emptyFinal = makePlan(empty, withProject, true);
    EXPECT_EQ(
        sortedRows(runPlan(emptyFinal, false, "17179869184")),
        sortedRows(runPlan(emptyFinal, true, "17179869184")));
  }
}

// Plans that are not a supported prefix ROLLUP keep the Expand operator and
// give the same result with the switch on; a supported one loses it.
TEST_F(RollupPartialAggTest, fusedRejected) {
  auto input = makeRowVector(
      {"c1", "c2", "x", "y"},
      {
          makeNullableFlatVector<std::string>(
              {"a", "a", "b", std::nullopt, "a"}),
          makeNullableFlatVector<int32_t>({1, 1, 2, 3, std::nullopt}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
          makeFlatVector<double>({1.5, 2.5, 3.5, 4.5, 5.5}),
      });
  const std::vector<std::vector<std::string>> prefixSets = {
      {"c1", "c2", "0 as gid", "x", "y"},
      {"c1", "null", "1", "x", "y"},
      {"null", "null", "3", "x", "y"}};
  const std::vector<std::vector<std::string>> nonPrefixSets = {
      {"c1", "c2", "0 as gid", "x", "y"},
      {"c1", "null", "1", "x", "y"},
      {"null", "c2", "2", "x", "y"}};
  auto makePlan = [&](const std::vector<std::vector<std::string>>& sets,
                      const std::vector<std::string>& aggregates) {
    return PlanBuilder()
        .values({input})
        .expand(sets)
        .partialAggregation({"c1", "c2", "gid"}, aggregates)
        .planNode();
  };
  auto run = [&](const core::PlanNodePtr& plan, bool fused, bool& hasExpand) {
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .config(
                          core::QueryConfig::kRollupPartialAggregationEnabled,
                          fused ? "true" : "false")
                      .copyResults(pool(), task);
    hasExpand = hasOperator(task, "Expand");
    return result;
  };
  struct Case {
    core::PlanNodePtr plan;
    bool expectFused;
  };
  const std::vector<Case> cases = {
      {makePlan(prefixSets, {"sum(x)"}), true},
      {makePlan(prefixSets, {"count(x)", "min(x)", "max(x)", "avg(x)"}), true},
      {makePlan(prefixSets, {"count(y)"}), true},
      // A floating point sum, or one such aggregate among supported ones.
      {makePlan(prefixSets, {"sum(y)"}), false},
      {makePlan(prefixSets, {"sum(x)", "avg(y)"}), false},
      {makePlan(prefixSets, {"arbitrary(x)"}), false},
      {makePlan(nonPrefixSets, {"sum(x)"}), false},
  };
  for (const auto& testCase : cases) {
    bool originalHasExpand = false;
    bool fusedHasExpand = false;
    auto original = run(testCase.plan, false, originalHasExpand);
    auto fused = run(testCase.plan, true, fusedHasExpand);
    EXPECT_TRUE(originalHasExpand);
    EXPECT_EQ(fusedHasExpand, !testCase.expectFused);
    EXPECT_EQ(sortedRows(original), sortedRows(fused));
  }
}

} // namespace
} // namespace facebook::velox::exec::test
