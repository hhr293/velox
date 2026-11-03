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
#include "velox/exec/RollupAggregation.h"

#include <atomic>
#include <numeric>
#include <unordered_set>

#include "velox/exec/AggregateInfo.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorHasher.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {
namespace {

// One Expand output column across all projections of the Expand node.
struct KeyColumn {
  // Input channel the column passes through.
  std::optional<column_index_t> channel;
  // Per projection: passed through (true) or replaced by a null (false).
  std::vector<bool> kept;
  // Per projection: the constant, for a grouping set id column.
  std::vector<std::shared_ptr<const core::ConstantTypedExpr>> constants;
  bool isGroupId{true};
};

// Returns std::nullopt unless every projection of 'column' is a field access
// to the same input channel or a null constant, or every projection is a
// non-null constant (a grouping set id).
std::optional<KeyColumn> analyzeColumn(
    const core::ExpandNode& expandNode,
    column_index_t column) {
  KeyColumn result;
  for (const auto& projection : expandNode.projections()) {
    const auto& expr = projection[column];
    if (auto field = core::TypedExprs::asFieldAccess(expr)) {
      const auto channel =
          expandNode.inputType()->getChildIdxIfExists(field->name());
      if (!channel.has_value() ||
          (result.channel.has_value() && result.channel != channel)) {
        return std::nullopt;
      }
      result.channel = channel;
      result.kept.push_back(true);
      result.constants.push_back(nullptr);
      result.isGroupId = false;
    } else if (auto constant = core::TypedExprs::asConstant(expr)) {
      result.kept.push_back(false);
      result.constants.push_back(constant);
      if (constant->isNull()) {
        result.isGroupId = false;
      }
    } else {
      return std::nullopt;
    }
  }
  if (result.isGroupId) {
    return result;
  }
  // A key that is a mix of fields and non-null constants is not a ROLLUP key.
  for (auto i = 0; i < result.kept.size(); ++i) {
    if (!result.kept[i] && !result.constants[i]->isNull()) {
      return std::nullopt;
    }
  }
  return result;
}

// Returns the function that merges the intermediate results of the aggregate
// function 'name' (which may carry the "_partial" suffix), or std::nullopt if
// 'name' is not one of the supported order insensitive aggregates. Only the
// exact names are accepted: a user defined function that merely ends in "_sum"
// need not have the semantics of sum.
std::optional<std::string> mergeFunctionName(std::string name) {
  constexpr std::string_view kPartialSuffix = "_partial";
  if (name.size() > kPartialSuffix.size() && name.ends_with(kPartialSuffix)) {
    name.resize(name.size() - kPartialSuffix.size());
  }
  for (const std::string_view function :
       {"sum", "min", "max", "avg", "count"}) {
    if (name == function) {
      return name;
    }
  }
  return std::nullopt;
}

// Floating point inputs are excluded because merging the levels changes the
// order of the additions.
bool isSupportedInputType(
    const std::string& mergeFunction,
    const TypePtr& type) {
  if (mergeFunction.ends_with("count")) {
    return true;
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return true;
    default:
      return type->isDecimal();
  }
}

void collectFieldNames(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& names) {
  if (auto field = core::TypedExprs::asFieldAccess(expr);
      field != nullptr && field->isInputColumn()) {
    names.insert(field->name());
    return;
  }
  for (const auto& input : expr->inputs()) {
    collectFieldNames(input, names);
  }
}

// Returns how to compute the input of 'aggregate' from the input of the Expand
// node and which function merges its intermediate results, or std::nullopt if
// the aggregate cannot be cascaded. The argument is computed once from the
// input of the Expand node, so it may only depend on columns that every
// projection of the Expand node passes through, and it must be deterministic.
std::optional<RollupAggregation::AggregateLayout> analyzeAggregate(
    const core::AggregationNode::Aggregate& aggregate,
    const core::ExpandNode& expandNode,
    const core::ProjectNode* projectNode,
    size_t numAppendedArguments,
    DriverCtx* driverCtx) {
  const auto mergeFunction = mergeFunctionName(aggregate.call->name());
  if (!mergeFunction.has_value() || aggregate.mask != nullptr ||
      aggregate.distinct || !aggregate.sortingKeys.empty() ||
      aggregate.rawInputTypes.size() != 1 ||
      !isSupportedInputType(*mergeFunction, aggregate.rawInputTypes[0])) {
    return std::nullopt;
  }
  const auto& inputs = aggregate.call->inputs();
  if (inputs.size() != 1) {
    return std::nullopt;
  }
  auto argument = core::TypedExprs::asFieldAccess(inputs[0]);
  if (argument == nullptr) {
    return std::nullopt;
  }

  core::TypedExprPtr argumentExpr = argument;
  if (projectNode != nullptr) {
    const auto index =
        projectNode->outputType()->getChildIdxIfExists(argument->name());
    if (!index.has_value()) {
      return std::nullopt;
    }
    argumentExpr = projectNode->projections()[*index];
  }
  std::unordered_set<std::string> argumentFields;
  collectFieldNames(argumentExpr, argumentFields);
  std::unordered_map<std::string, core::TypedExprPtr> inputNames;
  std::optional<column_index_t> argumentChannel;
  for (const auto& name : argumentFields) {
    const auto column = expandNode.outputType()->getChildIdxIfExists(name);
    if (!column.has_value()) {
      return std::nullopt;
    }
    const auto analysis = analyzeColumn(expandNode, *column);
    if (!analysis.has_value() || analysis->isGroupId ||
        !analysis->channel.has_value() ||
        std::find(analysis->kept.begin(), analysis->kept.end(), false) !=
            analysis->kept.end()) {
      return std::nullopt;
    }
    argumentChannel = analysis->channel;
    inputNames[name] = std::make_shared<core::FieldAccessTypedExpr>(
        expandNode.inputType()->childAt(*analysis->channel),
        expandNode.inputType()->nameOf(*analysis->channel));
  }

  RollupAggregation::AggregateLayout layout;
  layout.mergeFunctionName = *mergeFunction;
  if (core::TypedExprs::asFieldAccess(argumentExpr) != nullptr) {
    if (!argumentChannel.has_value()) {
      return std::nullopt;
    }
    layout.argumentChannel = *argumentChannel;
    return layout;
  }
  layout.argument = argumentExpr->rewriteInputNames(inputNames);
  // The name must be unique: drivers and tasks that share the query context
  // may analyze plans at the same time.
  static std::atomic<int64_t> analysisId{0};
  auto analysisPool = driverCtx->task->queryCtx()->pool()->addLeafChild(
      fmt::format("rollup_analysis_{}", analysisId++));
  core::ExecCtx execCtx(analysisPool.get(), driverCtx->task->queryCtx().get());
  ExprSet exprSet({layout.argument}, &execCtx);
  if (!exprSet.exprs()[0]->isDeterministic()) {
    return std::nullopt;
  }
  layout.argumentChannel =
      expandNode.inputType()->size() + numAppendedArguments;
  return layout;
}

std::optional<RollupAggregation::Layout> analyze(
    const std::shared_ptr<const core::ExpandNode>& expandNode,
    const std::shared_ptr<const core::ProjectNode>& projectNode,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode,
    DriverCtx* driverCtx) {
  using Step = core::AggregationNode::Step;
  const core::PlanNode* aggregationSource = projectNode
      ? static_cast<const core::PlanNode*>(projectNode.get())
      : expandNode.get();
  if (aggregationNode->step() != Step::kPartial ||
      aggregationNode->isPreGrouped() || aggregationNode->ignoreNullKeys() ||
      !aggregationNode->globalGroupingSets().empty() ||
      aggregationNode->groupId().has_value() ||
      aggregationNode->sources()[0].get() != aggregationSource ||
      (projectNode && projectNode->sources()[0].get() != expandNode.get()) ||
      aggregationNode->aggregates().empty()) {
    return std::nullopt;
  }

  const auto& expandType = expandNode->outputType();
  // Maps a grouping key of the aggregation to the Expand output it renames.
  auto resolveKey = [&](const std::string& name) -> std::optional<std::string> {
    if (!projectNode) {
      return name;
    }
    const auto index = projectNode->outputType()->getChildIdxIfExists(name);
    if (!index.has_value()) {
      return std::nullopt;
    }
    auto field =
        core::TypedExprs::asFieldAccess(projectNode->projections()[*index]);
    if (field == nullptr || !field->isInputColumn()) {
      return std::nullopt;
    }
    return field->name();
  };

  RollupAggregation::Layout layout;
  size_t numAppendedArguments = 0;
  for (const auto& aggregate : aggregationNode->aggregates()) {
    auto aggregateLayout = analyzeAggregate(
        aggregate,
        *expandNode,
        projectNode.get(),
        numAppendedArguments,
        driverCtx);
    if (!aggregateLayout.has_value()) {
      return std::nullopt;
    }
    if (aggregateLayout->argument != nullptr) {
      ++numAppendedArguments;
    }
    layout.aggregates.push_back(std::move(aggregateLayout.value()));
  }

  const auto numProjections = expandNode->projections().size();
  std::vector<KeyColumn> rollupKeys;
  for (const auto& key : aggregationNode->groupingKeys()) {
    const auto expandName = resolveKey(key->name());
    if (!expandName.has_value()) {
      return std::nullopt;
    }
    const auto column = expandType->getChildIdxIfExists(*expandName);
    if (!column.has_value()) {
      return std::nullopt;
    }
    auto analysis = analyzeColumn(*expandNode, *column);
    if (!analysis.has_value()) {
      return std::nullopt;
    }
    if (analysis->isGroupId) {
      if (!layout.groupIds.empty() || !key->type()->kindEquals(BIGINT())) {
        return std::nullopt;
      }
      layout.outputKeys.push_back(RollupAggregation::kGroupIdKey);
      // Filled in below, once the level of each projection is known.
      layout.groupIds = analysis->constants;
    } else {
      if (!analysis->channel.has_value()) {
        return std::nullopt;
      }
      layout.outputKeys.push_back(rollupKeys.size());
      layout.keyChannels.push_back(*analysis->channel);
      rollupKeys.push_back(std::move(analysis.value()));
    }
  }
  const auto numKeys = rollupKeys.size();
  if (layout.groupIds.size() != numProjections || numKeys == 0 ||
      numProjections != numKeys + 1) {
    return std::nullopt;
  }
  layout.numKeys = numKeys;

  // Each projection must keep a prefix of the keys, and every prefix length
  // must occur once.
  std::vector<std::shared_ptr<const core::ConstantTypedExpr>> groupIds(
      numKeys + 1);
  for (auto projection = 0; projection < numProjections; ++projection) {
    size_t numKept = 0;
    while (numKept < numKeys && rollupKeys[numKept].kept[projection]) {
      ++numKept;
    }
    for (auto key = numKept; key < numKeys; ++key) {
      if (rollupKeys[key].kept[projection]) {
        return std::nullopt;
      }
    }
    if (groupIds[numKept] != nullptr || layout.groupIds[projection]->isNull()) {
      return std::nullopt;
    }
    groupIds[numKept] = layout.groupIds[projection];
  }
  layout.groupIds = std::move(groupIds);
  return layout;
}

core::FieldAccessTypedExprPtr makeField(
    const RowTypePtr& type,
    uint32_t index) {
  return std::make_shared<core::FieldAccessTypedExpr>(
      type->childAt(index), type->nameOf(index));
}

// Creates the node that toAggregateInfo() needs to set up a grouping set. It is
// never executed.
std::shared_ptr<const core::AggregationNode> makeLevelNode(
    const core::AggregationNode& original,
    bool isRawInput,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    const std::vector<column_index_t>& argumentChannels,
    const std::vector<std::string>& functionNames,
    memory::MemoryPool* pool) {
  std::vector<core::FieldAccessTypedExprPtr> keys;
  for (auto channel : keyChannels) {
    keys.push_back(makeField(inputType, channel));
  }
  std::vector<core::AggregationNode::Aggregate> aggregates;
  for (auto i = 0; i < functionNames.size(); ++i) {
    const auto& originalAggregate = original.aggregates()[i];
    core::AggregationNode::Aggregate aggregate;
    aggregate.rawInputTypes = originalAggregate.rawInputTypes;
    aggregate.call = std::make_shared<core::CallTypedExpr>(
        originalAggregate.call->type(),
        std::vector<core::TypedExprPtr>{
            makeField(inputType, argumentChannels[i])},
        functionNames[i]);
    aggregates.push_back(std::move(aggregate));
  }
  auto source = std::make_shared<core::ValuesNode>(
      "rollup_level_source",
      std::vector<RowVectorPtr>{
          BaseVector::create<RowVector>(inputType, 0, pool)});
  return std::make_shared<core::AggregationNode>(
      "rollup_level",
      isRawInput ? core::AggregationNode::Step::kPartial
                 : core::AggregationNode::Step::kIntermediate,
      keys,
      std::vector<core::FieldAccessTypedExprPtr>{},
      original.aggregateNames(),
      std::move(aggregates),
      /*ignoreNullKeys=*/false,
      /*noGroupsSpanBatches=*/false,
      std::move(source));
}

} // namespace

std::unique_ptr<RollupAggregation> RollupAggregation::tryCreate(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode,
    const std::shared_ptr<const core::ProjectNode>& projectNode,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode) {
  auto layout = analyze(expandNode, projectNode, aggregationNode, driverCtx);
  if (!layout.has_value()) {
    return nullptr;
  }
  return std::unique_ptr<RollupAggregation>(new RollupAggregation(
      operatorId,
      driverCtx,
      expandNode,
      aggregationNode,
      std::move(layout.value())));
}

RollupAggregation::RollupAggregation(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode,
    Layout layout)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          OperatorType::kPartialAggregation),
      aggregationNode_(aggregationNode),
      inputType_(expandNode->inputType()),
      layout_(std::move(layout)) {
  auto names = inputType_->names();
  auto types = inputType_->children();
  for (const auto& aggregate : layout_.aggregates) {
    if (aggregate.argument != nullptr) {
      names.push_back(fmt::format("rollup_argument_{}", names.size()));
      types.push_back(aggregate.argument->type());
      argumentExprList_.push_back(aggregate.argument);
    }
  }
  inputType_ = ROW(std::move(names), std::move(types));
}

namespace {
// Tells the aggregate functions where their accumulators are in the rows of
// 'rows'.
void initializeAggregates(
    const std::vector<AggregateInfo>& aggregates,
    RowContainer& rows,
    int32_t numKeys) {
  int i = 0;
  for (auto& aggregate : aggregates) {
    auto& function = aggregate.function;
    function->setAllocator(&rows.stringAllocator());
    const auto rowColumn = rows.columnAt(numKeys + i);
    function->setOffsets(
        rowColumn.offset(),
        rowColumn.nullByte(),
        rowColumn.nullMask(),
        rowColumn.initializedByte(),
        rowColumn.initializedMask(),
        rows.rowSizeOffset());
    ++i;
  }
}
} // namespace

void RollupAggregation::initialize() {
  Operator::initialize();
  VELOX_CHECK(pool()->trackUsage());

  if (!argumentExprList_.empty()) {
    argumentExprs_ =
        std::make_unique<ExprSet>(argumentExprList_, operatorCtx_->execCtx());
  }

  const auto numKeys = layout_.numKeys;
  levels_.resize(numKeys + 1);
  currentLevel_ = -1;
  maxPartialMemoryUsage_ = operatorCtx_->driverCtx()
                               ->queryConfig()
                               .maxPartialAggregationMemoryUsage();
  for (auto key = 0; key < numKeys; ++key) {
    keyTypes_.push_back(inputType_->childAt(layout_.keyChannels[key]));
  }

  std::vector<RowTypePtr> levelOutputTypes(numKeys + 1);
  std::shared_ptr<core::ExpressionEvaluator> expressionEvaluator;
  for (int64_t level = numKeys; level >= 0; --level) {
    const bool isRawInput = level == numKeys;
    const auto& levelInputType =
        isRawInput ? inputType_ : levelOutputTypes[level + 1];
    std::vector<column_index_t> keyChannels;
    if (isRawInput) {
      keyChannels = layout_.keyChannels;
    } else {
      keyChannels.resize(level);
      std::iota(keyChannels.begin(), keyChannels.end(), 0);
    }
    std::vector<column_index_t> argumentChannels;
    std::vector<std::string> functionNames;
    for (auto i = 0; i < layout_.aggregates.size(); ++i) {
      const auto& aggregate = layout_.aggregates[i];
      argumentChannels.push_back(
          isRawInput ? aggregate.argumentChannel
                     : static_cast<column_index_t>(level + 1 + i));
      functionNames.push_back(
          isRawInput ? aggregationNode_->aggregates()[i].call->name()
                     : aggregate.mergeFunctionName);
    }

    auto node = makeLevelNode(
        *aggregationNode_,
        isRawInput,
        levelInputType,
        keyChannels,
        argumentChannels,
        functionNames,
        pool());

    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (auto key = 0; key < level; ++key) {
      names.push_back(fmt::format("rollup_key_{}", key));
      types.push_back(levelInputType->childAt(keyChannels[key]));
    }
    for (auto i = 0; i < layout_.aggregates.size(); ++i) {
      names.push_back(fmt::format("rollup_aggregate_{}", i));
      types.push_back(node->outputType()->childAt(level + i));
    }
    levelOutputTypes[level] = ROW(std::move(names), std::move(types));

    auto& tree = levels_[level];
    tree.aggregates =
        toAggregateInfo(*node, *operatorCtx_, level, expressionEvaluator);
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    hashers.push_back(VectorHasher::create(BIGINT(), 0));
    std::vector<std::string> probeNames{"parent"};
    std::vector<TypePtr> probeTypes{BIGINT()};
    if (level > 0) {
      hashers.push_back(VectorHasher::create(keyTypes_[level - 1], 1));
      probeNames.push_back("key");
      probeTypes.push_back(keyTypes_[level - 1]);
    }
    tree.probeType = ROW(std::move(probeNames), std::move(probeTypes));
    std::vector<Accumulator> accumulators;
    for (const auto& aggregate : tree.aggregates) {
      accumulators.emplace_back(
          aggregate.function.get(), aggregate.intermediateType);
    }
    tree.table = HashTable<false>::createForAggregation(
        std::move(hashers), accumulators, pool());
    auto& rows = *tree.table->rows();
    const auto numTableKeys = rows.keyTypes().size();
    initializeAggregates(tree.aggregates, rows, numTableKeys);
    tree.lookup = std::make_unique<HashLookup>(tree.table->hashers(), pool());
    tree.parentOffset = rows.columnAt(0).offset();
    tree.aggregateOffset = rows.columnAt(numTableKeys).offset();
  }
  allSupportToIntermediate_ = true;
  for (const auto& aggregate : levels_[numKeys].aggregates) {
    allSupportToIntermediate_ &= aggregate.function->supportsToIntermediate();
  }
  resetTree();
}

void RollupAggregation::resetTree() {
  for (auto& level : levels_) {
    level.table->clear(true);
  }
  auto& root = levels_[0];
  auto zero = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool());
  zero->set(0, 0);
  auto probe = std::make_shared<RowVector>(
      pool(), root.probeType, nullptr, 1, std::vector<VectorPtr>{zero});
  SelectivityVector rows(1);
  root.table->prepareForGroupProbe(
      *root.lookup, probe, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
  root.table->groupProbe(
      *root.lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
  char** groups = root.lookup->hits.data();
  for (auto& aggregate : root.aggregates) {
    aggregate.function->initializeNewGroups(groups, root.lookup->newGroups);
  }
  rootRow_ = groups[0];
  hasInput_ = false;
}

int64_t RollupAggregation::treeBytes() const {
  int64_t bytes = 0;
  for (const auto& level : levels_) {
    bytes += level.table->allocatedBytes();
  }
  return bytes;
}

bool RollupAggregation::needsInput() const {
  return !noMoreInput_ && !draining_ && pending_.empty();
}

void RollupAggregation::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  if (argumentExprs_ != nullptr) {
    input = appendArgument(input);
  }
  const auto size = input->size();
  if (size == 0) {
    return;
  }
  if (abandoned_) {
    passThrough(input);
    return;
  }
  hasInput_ = true;
  numInputRows_ += size;
  if (parentVector_ == nullptr || parentVector_->size() < size) {
    parentVector_ =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), size, pool());
  }
  parentVector_->resize(size);
  auto* parents = parentVector_->mutableRawValues();
  std::fill(parents, parents + size, reinterpret_cast<int64_t>(rootRow_));

  const auto numKeys = layout_.numKeys;
  for (auto level = 1; level <= numKeys; ++level) {
    auto& tree = levels_[level];
    auto probe = std::make_shared<RowVector>(
        pool(),
        tree.probeType,
        nullptr,
        size,
        std::vector<VectorPtr>{
            parentVector_, input->childAt(layout_.keyChannels[level - 1])});
    SelectivityVector rows(size);
    tree.table->prepareForGroupProbe(
        *tree.lookup,
        probe,
        rows,
        BaseHashTable::kNoSpillInputStartPartitionBit);
    tree.table->groupProbe(
        *tree.lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
    char** groups = tree.lookup->hits.data();
    if (!tree.lookup->newGroups.empty()) {
      for (auto& aggregate : tree.aggregates) {
        aggregate.function->initializeNewGroups(groups, tree.lookup->newGroups);
      }
    }
    if (level == numKeys) {
      for (auto& aggregate : tree.aggregates) {
        std::vector<VectorPtr> args{input->childAt(aggregate.inputs[0])};
        aggregate.function->addRawInput(groups, rows, args, false);
      }
    } else {
      for (vector_size_t i = 0; i < size; ++i) {
        parents[i] = reinterpret_cast<int64_t>(groups[i]);
      }
    }
  }
  if (treeBytes() > maxPartialMemoryUsage_) {
    startFlush();
  }
}

RowVectorPtr RollupAggregation::appendArgument(const RowVectorPtr& input) {
  const auto size = input->size();
  SelectivityVector rows(size);
  EvalCtx evalCtx(operatorCtx_->execCtx(), argumentExprs_.get(), input.get());
  std::vector<VectorPtr> results;
  argumentExprs_->eval(rows, evalCtx, results);
  auto columns = input->children();
  for (auto& result : results) {
    columns.push_back(std::move(result));
  }
  return std::make_shared<RowVector>(
      pool(), inputType_, nullptr, size, std::move(columns));
}

void RollupAggregation::rollUp() {
  constexpr int32_t kBatch = 1024;
  char* rows[kBatch];
  char* parents[kBatch];
  for (auto level = layout_.numKeys; level >= 1; --level) {
    auto& child = levels_[level];
    auto& parent = levels_[level - 1];
    RowContainerIterator iterator;
    while (auto count =
               child.table->rows()->listRows(&iterator, kBatch, rows)) {
      for (auto i = 0; i < count; ++i) {
        parents[i] = reinterpret_cast<char*>(
            *reinterpret_cast<int64_t*>(rows[i] + child.parentOffset));
      }
      // The parent rows are in no particular order: fetch them while the
      // intermediate results of the children are extracted.
      for (auto i = 0; i < count; ++i) {
        __builtin_prefetch(parents[i] + parent.aggregateOffset, 1, 3);
      }
      SelectivityVector all(count);
      for (auto i = 0; i < child.aggregates.size(); ++i) {
        VectorPtr intermediate = BaseVector::create(
            child.aggregates[i].intermediateType, count, pool());
        child.aggregates[i].function->extractAccumulators(
            rows, count, &intermediate);
        parent.aggregates[i].function->addIntermediateResults(
            parents, all, {intermediate}, false);
      }
    }
  }
}

void RollupAggregation::startFlush() {
  // If more than this many of the rows the unfused plan would aggregate end up
  // as groups of their own, giving up on the aggregation is cheaper. Same
  // limit as HashAggregation applies to a partial aggregation at full memory.
  constexpr int32_t kAbandonMinPct = 40;
  draining_ = true;
  levelStarted_ = false;
  if (hasInput_) {
    if (!noMoreInput_ && allSupportToIntermediate_) {
      int64_t nodes = 0;
      for (const auto& level : levels_) {
        nodes += level.table->rows()->numRows();
      }
      // The unfused plan aggregates numKeys + 1 rows for every input row.
      const int64_t expanded = numInputRows_ * (layout_.numKeys + 1);
      abandonAfterFlush_ = 100 * nodes / expanded > kAbandonMinPct;
    }
    rollUp();
    currentLevel_ = layout_.numKeys;
  } else {
    currentLevel_ = -1;
  }
  numInputRows_ = 0;
  if (!noMoreInput_) {
    ++numFlushes_;
  }
}

void RollupAggregation::passThrough(const RowVectorPtr& input) {
  const auto size = input->size();
  SelectivityVector rows(size);
  std::vector<VectorPtr> intermediates;
  for (auto& aggregate : levels_[layout_.numKeys].aggregates) {
    VectorPtr result =
        BaseVector::create(aggregate.intermediateType, size, pool());
    std::vector<VectorPtr> args{input->childAt(aggregate.inputs[0])};
    aggregate.function->toIntermediate(rows, args, result);
    intermediates.push_back(std::move(result));
  }
  for (int64_t level = layout_.numKeys; level >= 0; --level) {
    std::vector<VectorPtr> columns;
    columns.reserve(outputType_->size());
    for (auto column = 0; column < layout_.outputKeys.size(); ++column) {
      const auto key = layout_.outputKeys[column];
      if (key == kGroupIdKey) {
        columns.push_back(
            BaseVector::wrapInConstant(
                size, 0, layout_.groupIds[level]->toConstantVector(pool())));
      } else if (key < level) {
        columns.push_back(input->childAt(layout_.keyChannels[key]));
      } else {
        columns.push_back(
            BaseVector::createNullConstant(
                outputType_->childAt(column), size, pool()));
      }
    }
    for (const auto& intermediate : intermediates) {
      columns.push_back(intermediate);
    }
    pending_.push_back(
        std::make_shared<RowVector>(
            pool(), outputType_, nullptr, size, std::move(columns)));
  }
}

RowVectorPtr RollupAggregation::makeOutput(
    column_index_t level,
    char** rows,
    int32_t count) {
  // The ancestors of the nodes, by level.
  std::vector<std::vector<char*>> ancestors(level + 1);
  ancestors[level].assign(rows, rows + count);
  for (auto l = level; l >= 2; --l) {
    ancestors[l - 1].resize(count);
    for (auto i = 0; i < count; ++i) {
      ancestors[l - 1][i] = reinterpret_cast<char*>(*reinterpret_cast<int64_t*>(
          ancestors[l][i] + levels_[l].parentOffset));
    }
  }
  std::vector<VectorPtr> columns;
  columns.reserve(outputType_->size());
  for (auto column = 0; column < layout_.outputKeys.size(); ++column) {
    const auto key = layout_.outputKeys[column];
    if (key == kGroupIdKey) {
      columns.push_back(
          BaseVector::wrapInConstant(
              count, 0, layout_.groupIds[level]->toConstantVector(pool())));
    } else if (key < level) {
      VectorPtr keyVector = BaseVector::create(keyTypes_[key], count, pool());
      levels_[key + 1].table->rows()->extractColumn(
          ancestors[key + 1].data(), count, 1, keyVector);
      columns.push_back(std::move(keyVector));
    } else {
      columns.push_back(
          BaseVector::createNullConstant(
              outputType_->childAt(column), count, pool()));
    }
  }
  for (auto i = 0; i < layout_.aggregates.size(); ++i) {
    VectorPtr aggregate = BaseVector::create(
        outputType_->childAt(layout_.outputKeys.size() + i), count, pool());
    levels_[level].aggregates[i].function->extractAccumulators(
        rows, count, &aggregate);
    columns.push_back(std::move(aggregate));
  }
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, count, std::move(columns));
}

RowVectorPtr RollupAggregation::getOutput() {
  if (!pending_.empty()) {
    auto output = std::move(pending_.front());
    pending_.pop_front();
    return output;
  }
  if (!draining_) {
    if (abandoned_) {
      finished_ = noMoreInput_;
      return nullptr;
    }
    if (!noMoreInput_ || finished_) {
      return nullptr;
    }
    startFlush();
  }
  const auto& queryConfig = operatorCtx_->driverCtx()->queryConfig();
  while (currentLevel_ >= 0) {
    auto& tree = levels_[currentLevel_];
    if (!levelStarted_) {
      iterator_.reset();
      levelStarted_ = true;
    }
    // The root outputs exactly one row.
    const int32_t maxRows =
        currentLevel_ == 0 ? 1 : queryConfig.preferredOutputBatchRows();
    outputRows_.resize(maxRows);
    const auto count =
        tree.table->rows()->listRows(&iterator_, maxRows, outputRows_.data());
    if (count == 0) {
      --currentLevel_;
      levelStarted_ = false;
      continue;
    }
    return makeOutput(currentLevel_, outputRows_.data(), count);
  }
  resetTree();
  draining_ = false;
  if (abandonAfterFlush_) {
    abandoned_ = true;
    abandonAfterFlush_ = false;
    stats_.wlock()->addRuntimeStat("rollup.abandoned", RuntimeCounter(1));
  }
  finished_ = noMoreInput_;
  if (finished_) {
    stats_.wlock()->addRuntimeStat(
        "rollup.numFlushes", RuntimeCounter(numFlushes_));
  }
  return nullptr;
}

bool RollupAggregation::isFinished() {
  return finished_ && pending_.empty();
}

} // namespace facebook::velox::exec
