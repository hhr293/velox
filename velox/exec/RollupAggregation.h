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
#pragma once

#include <deque>

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateInfo.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

/// Fuses an Expand node that produces a complete prefix ROLLUP with the
/// partial aggregation above it. A ROLLUP over K keys has K + 1 grouping levels
/// and every level is a prefix of the next one, so the groups form a prefix
/// tree: a node of level j is identified by its parent and the value of key j.
/// The operator keeps one hash table per level, keyed by (parent row, key), and
/// adds every input row only to its leaf. When the tree is output, every node
/// is merged into its parent from the leaves up, so no coarser level is ever
/// hashed on the full key. The output has the schema and content of the
/// unfused partial aggregation, up to row order.
///
/// Supports sum, min, max, avg and count aggregates over integer or DECIMAL
/// inputs, without mask, DISTINCT or ordering. Flushes the tree when it reaches
/// max_partial_aggregation_memory. Like HashAggregation, it gives up
/// aggregating when a flush shows that the groups are more than 40% of the rows
/// the unfused plan would aggregate (provided every aggregate supports
/// toIntermediate): from then on it passes every input row on as the
/// intermediate result of a group of its own.
class RollupAggregation : public Operator {
 public:
  /// Returns an operator for 'expandNode' followed by an optional
  /// 'projectNode' and 'aggregationNode', or nullptr if the nodes do not form a
  /// supported prefix ROLLUP, in which case the caller keeps the unfused plan.
  /// The aggregate arguments computed by the projection are evaluated once per
  /// input row of the Expand node.
  static std::unique_ptr<RollupAggregation> tryCreate(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExpandNode>& expandNode,
      const std::shared_ptr<const core::ProjectNode>& projectNode,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode);

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  /// Describes how to compute the input of one aggregate.
  struct AggregateLayout {
    // Input channel of the aggregate argument. When 'argument' is set, the
    // channel of the column appended to the input.
    column_index_t argumentChannel;
    // Expression over the input of the Expand node that computes the aggregate
    // argument, or nullptr if the argument is an input column.
    core::TypedExprPtr argument;
    // Name of the function that merges the intermediate results of the
    // aggregate.
    std::string mergeFunctionName;
  };

  /// Describes how the grouping keys and aggregates of the aggregation map to
  /// the input of the Expand node.
  struct Layout {
    // Number of ROLLUP keys. The levels are numKeys, ..., 0.
    column_index_t numKeys;
    // Input channel of each ROLLUP key, in prefix order.
    std::vector<column_index_t> keyChannels;
    std::vector<AggregateLayout> aggregates;
    // For each grouping key of the aggregation: the index of the ROLLUP key,
    // or kGroupIdKey for the grouping set id.
    std::vector<column_index_t> outputKeys;
    // Grouping set id of each level, indexed by the number of kept keys.
    std::vector<std::shared_ptr<const core::ConstantTypedExpr>> groupIds;
  };

  static constexpr column_index_t kGroupIdKey = ~column_index_t{0};

 private:
  RollupAggregation(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExpandNode>& expandNode,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode,
      Layout layout);

  void initialize() override;

  // Returns 'input' with the aggregate argument appended as the last column.
  RowVectorPtr appendArgument(const RowVectorPtr& input);

  // The levels form a prefix tree: a node of level j is a row of the hash table
  // of level j, keyed by (pointer to its parent row, value of key j). Level 0
  // has the single root node. Rows are only aggregated into the leaves (level
  // numKeys); roll-up adds every node to its parent when the tree is output.
  struct Level {
    std::unique_ptr<HashTable<false>> table;
    std::unique_ptr<HashLookup> lookup;
    std::vector<AggregateInfo> aggregates;
    // Offset of the parent row pointer in a row, and of the first aggregate.
    int32_t parentOffset{0};
    int32_t aggregateOffset{0};
    // Type of the input of the probe: (parent pointer, key).
    RowTypePtr probeType;
  };

  // Clears the tree and adds the root node.
  void resetTree();

  // Adds every node to its parent, from the leaves to the root.
  void rollUp();

  // Starts outputting all the levels, either because the tree is full or at
  // the end of input.
  void startFlush();

  // After the partial aggregation was abandoned: outputs every row of 'input'
  // as the intermediate result of a group of its own, for every level, like
  // the unfused plan does when the aggregation does not reduce the data.
  void passThrough(const RowVectorPtr& input);

  // Output of 'count' nodes of 'level'.
  RowVectorPtr makeOutput(column_index_t level, char** rows, int32_t count);

  int64_t treeBytes() const;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  RowTypePtr inputType_;
  Layout layout_;

  // The computed aggregate arguments, in the order they are appended to the
  // input, and their evaluator.
  std::vector<core::TypedExprPtr> argumentExprList_;
  std::unique_ptr<ExprSet> argumentExprs_;

  std::vector<TypePtr> keyTypes_;
  std::vector<Level> levels_;
  char* rootRow_{nullptr};
  FlatVectorPtr<int64_t> parentVector_;
  std::vector<char*> outputRows_;
  // Level being output, counted in kept keys. Negative once all levels are
  // done.
  int64_t currentLevel_{-1};
  // True while the levels are being output; no input is accepted meanwhile.
  bool draining_{false};
  bool finished_{false};
  bool levelStarted_{false};
  // True if rows were added since the tree was last cleared.
  bool hasInput_{false};
  // Rows added since the tree was last cleared.
  int64_t numInputRows_{0};
  // True if every aggregate can turn a raw input row into an intermediate
  // result, which abandoning the aggregation needs.
  bool allSupportToIntermediate_{false};
  // Set when a flush shows that the tree does not reduce the data; the
  // aggregation is abandoned once the flush is output.
  bool abandonAfterFlush_{false};
  bool abandoned_{false};
  std::deque<RowVectorPtr> pending_;
  int64_t maxPartialMemoryUsage_{0};
  int32_t numFlushes_{0};
  RowContainerIterator iterator_;
};

} // namespace facebook::velox::exec
