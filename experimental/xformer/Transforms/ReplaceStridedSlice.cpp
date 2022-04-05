// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace TFL StridedSlice with StridedSlice for XCore.
struct ReplaceStridedSlice
    : public PassWrapper<ReplaceStridedSlice, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct ReplaceStridedSlicePattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSliceOp,
                                PatternRewriter &rewriter) const override {

    // Extract args from the op
    auto inputType =
        stridedSliceOp.input().getType().dyn_cast<RankedTensorType>();

    auto beginValuesConstOp =
        dyn_cast<mlir::ConstantOp>(stridedSliceOp.begin().getDefiningOp());
    auto beginValues =
        beginValuesConstOp.value().template cast<DenseElementsAttr>();

    auto endValuesConstOp =
        dyn_cast<mlir::ConstantOp>(stridedSliceOp.end().getDefiningOp());
    auto endValues =
        endValuesConstOp.value().template cast<DenseElementsAttr>();

    auto stridesValuesConstOp =
        dyn_cast<mlir::ConstantOp>(stridedSliceOp.strides().getDefiningOp());
    auto stridesValues =
        stridesValuesConstOp.value().template cast<DenseElementsAttr>();

    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);
    auto inputDepth = inputType.getDimSize(3);
    auto begin_x = beginValues.template getValue<int32_t>({2});
    auto begin_y = beginValues.template getValue<int32_t>({1});
    auto end_x = endValues.template getValue<int32_t>({2});
    auto end_y = endValues.template getValue<int32_t>({1});
    auto stride_x = stridesValues.template getValue<int32_t>({2});
    auto stride_y = stridesValues.template getValue<int32_t>({1});

    auto image_geom =
        nn::ImageGeometry(inputHeight, inputWidth, static_cast<int>(inputDepth));
    
    int x_diff = end_x - begin_x;
    int y_diff = end_y - begin_y;
    auto window_geom = nn::WindowGeometry(
        {y_diff,
         x_diff, static_cast<int>(inputDepth)},
        {begin_y, begin_x}, {1, 1, 1}, {stride_y, stride_x});

    nn::ImToColValid::Params imToColParams(image_geom, window_geom,static_cast<int>(inputDepth));

    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();

    llvm::SmallVector<std::string> strParams;

    strParams.push_back(mfStr);

    std::string memcpyFnParam;

    memcpyFnParam = strParams[0];

    auto binaryObjectStridedSliceOp = rewriter.create<StridedSliceV3Op>(
        stridedSliceOp.getLoc(), stridedSliceOp.getType(),stridedSliceOp.input(),
        rewriter.getI32IntegerAttr(begin_x),
        rewriter.getI32IntegerAttr(begin_y),
        rewriter.getStringAttr(memcpyFnParam));
    rewriter.replaceOp(stridedSliceOp, binaryObjectStridedSliceOp.output());

    return success();
  }
};

void ReplaceStridedSlice::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceStridedSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceStridedSlice pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceStridedSlicePass() {
  return std::make_unique<ReplaceStridedSlice>();
}

static PassRegistration<ReplaceStridedSlice>
    pass("xcore-replace-StridedSlice",
         "Replace TFL StridedSlice with StridedSlice for XCore.");

} // namespace xcore
} // namespace mlir