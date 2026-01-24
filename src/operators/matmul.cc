#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Shape shapeA = inputs[0]->getDims();
        Shape shapeB = inputs[1]->getDims();

        if (shapeA.size() == 1) {
            shapeA = {1, shapeA[0]};
        }
        if (shapeB.size() == 1) {
            shapeB = {shapeB[0], 1};
        }
        if (transA) {
            std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
        }
        if (transB) {
            std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);
        }

        size_t rankA = shapeA.size();
        size_t rankB = shapeB.size();
        size_t rankO = std::max(rankA, rankB);
        Shape outputShape(rankO);
        outputShape[outputShape.size() - 1] = shapeB[outputShape.size() - 1];
        outputShape[outputShape.size() - 2] = shapeA[outputShape.size() - 2];
        for (size_t i = 0; i + 2 < outputShape.size(); i++) {
            ShapeElem a = (i < rankA - 2) ? shapeA[rankA - 3 - i] : 1;
            ShapeElem b = (i < rankB - 2) ? shapeB[rankB - 3 - i] : 1;

            if (a != b && a != 1 && b != 1) {
                throw Exception(
                    "ONNX broadcast failed: incompatible dimensions"
                );
            }
            outputShape[rankO - 3 - i] = std::max(a, b);
        }
        return {{outputShape}};
    }

} // namespace infini