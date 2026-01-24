//
// Created by flashzxi on 1/24/26.
//
#include "optimizer/rules.h"
#include "core/op_type.h"
#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini {

bool AntiTransposePattern::match(Operator op) {
    if (op->getPredecessors().size() != 1) {
        return false;
    }
    Operator preOp = op->getPredecessors()[0];
    if (op->getOpType() != OpType::Transpose || preOp->getOpType() != OpType::Transpose) {
        return false;
    }
    auto opAsTrans = std::dynamic_pointer_cast<TransposeObj>(op);
    auto preOpAsTrans = std::dynamic_pointer_cast<TransposeObj>(preOp);
    IT_ASSERT(opAsTrans != nullptr, "Impossible");
    IT_ASSERT(preOpAsTrans != nullptr, "Impossible");
    std::vector<int> opPermute = opAsTrans->getPermute();
    std::vector<int> preOpPermute = preOpAsTrans->getPermute();
    if (opPermute.size() != preOpPermute.size()) {
        return false;
    }
    for (size_t i = 0; i < opPermute.size(); ++i) {
        if (preOpPermute[opPermute[i]] != static_cast<int>(i)) {
            return false;
        }
    }

    return true;
}

void AntiTransposeRewriter::rewrite(Operator op, Graph graph) {
    IT_ASSERT(op->getPredecessors().size() == 1, "Impossible");
    Operator preOp = op->getPredecessors()[0];

    OpVec sucOps = op->getSuccessors();

    // prePreOp 直接接在sucOp 上
    IT_ASSERT(preOp->getInputs().size() == 1, "Impossible");
    auto input = preOp->getInputs(0);

    if (input) {
        // preOp 所有的input，后继删除本preOp，添加全部的sucOp
        input->removeTarget(preOp);
        for (const auto& suc: sucOps) {
            if (suc) {
                input->addTarget(suc);
                suc->removePredecessors(op);
            }
        }

        auto pred = input->getSource();
        if (pred) {
            pred->removeSuccessors(preOp);
            for (const auto& suc: sucOps) {
                if (suc) {
                    pred->addSuccessors(suc);
                    suc->addPredecessors(pred);
                }
            }
        }
    }

    for (auto& sucOp: sucOps) {
        if (sucOp) {
            const TensorVec &sucInputs = sucOp->getInputs();
            for (size_t i = 0; i < sucInputs.size(); ++i) {
                auto it = sucInputs[i];
                if (it.get() == op->getOutput().get()) {
                    sucOp->resetInput(i, input);
                }
            }
        }
    }

    graph->removeTensor(op->getOutput());
    graph->removeTensor(preOp->getOutput());
    graph->removeOperator(op);
    graph->removeOperator(preOp);
}

bool TransposeBeforeGEMMPattern::match(Operator op) {
    if (op == nullptr || op->getOpType() != OpType::MatMul) {
        return false;
    }
    TensorVec preds = op->getInputs();
    assert(preds.size() == 2);
    const Operator& child_left = preds[0]->getSource();
    const Operator& child_right = preds[1]->getSource();

    auto check_permute = [] (const vector<int>& permute) {
        if (permute.size() >= 2) {
            auto permute_cpy = permute;
            std::swap(permute_cpy[permute_cpy.size() - 1], permute_cpy[permute_cpy.size() - 2]);
            for (size_t i = 0; i < permute.size(); ++i) {
                if (permute_cpy[i] != static_cast<int>(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    };
    if (child_left && child_left->getOpType() == OpType::Transpose) {
        auto left_as_trans = std::dynamic_pointer_cast<TransposeObj>(child_left);
        std::vector<int> permute = left_as_trans->getPermute();
        if (check_permute(permute)) {
            return true;
        }
    }
    if (child_right && child_right->getOpType() == OpType::Transpose) {
        auto right_as_trans = std::dynamic_pointer_cast<TransposeObj>(child_right);
        std::vector<int> permute = right_as_trans->getPermute();
        if (check_permute(permute)) {
            return true;
        }
    }
    return false;
}

void TransposeBeforeGEMMRewriter::rewrite(Operator op, Graph graph) {
    TensorVec preds = op->getInputs();
    assert(preds.size() == 2);
    const Operator& child_left = preds[0]->getSource();
    const Operator& child_right = preds[1]->getSource();
    auto matmulObj = std::dynamic_pointer_cast<MatmulObj>(op);

    auto check_permute = [] (const vector<int>& permute) {
        if (permute.size() >= 2) {
            auto permute_cpy = permute;
            std::swap(permute_cpy[permute_cpy.size() - 1], permute_cpy[permute_cpy.size() - 2]);
            for (size_t i = 0; i < permute.size(); ++i) {
                if (static_cast<int>(i) != permute_cpy[i]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    };
    if (child_left && child_left->getOpType() == OpType::Transpose) {
        auto left_as_trans = std::dynamic_pointer_cast<TransposeObj>(child_left);
        std::vector<int> permute = left_as_trans->getPermute();
        if (check_permute(permute)) {
            matmulObj->setTransA(true);
            auto leftOp = matmulObj->getInputs()[0]->getSource();
            IT_ASSERT(leftOp->getInputs().size() == 1, "Impossible");
            auto pred = leftOp->getInputs(0);
            pred->removeTarget(leftOp);
            pred->addTarget(matmulObj);
            matmulObj->removePredecessors(leftOp);
            if (Operator predSource = pred->getSource()) {
                predSource->removeSuccessors(leftOp);
                predSource->addSuccessors(matmulObj);
                matmulObj->addPredecessors(predSource);
            }
            matmulObj->resetInput(0, pred);
            graph->removeTensor(leftOp->getOutput());
            graph->removeOperator(leftOp);
        }
    }
    if (child_right && child_right->getOpType() == OpType::Transpose) {
        auto right_as_trans = std::dynamic_pointer_cast<TransposeObj>(child_right);
        std::vector<int> permute = right_as_trans->getPermute();
        if (check_permute(permute)) {
            matmulObj->setTransB(true);
            auto rightOp = matmulObj->getInputs()[1]->getSource();
            IT_ASSERT(rightOp->getInputs().size() == 1, "Impossible");
            auto pred = rightOp->getInputs(0);
            pred->removeTarget(rightOp);
            pred->addTarget(matmulObj);
            matmulObj->removePredecessors(rightOp);
            if (Operator predSource = pred->getSource()) {
                predSource->removeSuccessors(rightOp);
                predSource->addSuccessors(matmulObj);
                matmulObj->addPredecessors(predSource);
            }
            matmulObj->resetInput(1, pred);
            graph->removeTensor(rightOp->getOutput());
            graph->removeOperator(rightOp);
        }
    }
}

REGISTER_RULE(AntiTransposePattern, AntiTransposeRewriter);
REGISTER_RULE(TransposeBeforeGEMMPattern, TransposeBeforeGEMMRewriter);
} // namespace infini