//
// Created by flashzxi on 1/24/26.
//

#pragma once
#include "core/ref.h"
#include "core/operator.h"
#include "core/common.h"

namespace infini {

template<typename T>
struct PointerCmp {
    bool operator()(const Ref<T>& ref1, const Ref<T>& ref2) const {
        return *ref1 < *ref2;
    }
};

// 所有的优化规则都抽象为Rule
// Rule包括 pattern 以及 替换方法
// 含义为match到pattern则执行替换方法
class Pattern {
public:
    virtual bool match(Operator op) = 0;
    virtual std::string name() const = 0;
    bool operator<(const Pattern& other) const {
        return name() < other.name();
    }

    virtual ~Pattern() = default;
};

class ReWriter {
public:
    virtual void rewrite(Operator op, Graph graph) = 0;

    virtual ~ReWriter() = default;
};

class AntiTransposePattern : public Pattern {
public:
    std::string name() const override {
        return "AntiTransposePattern";
    }

    bool match(Operator op) override;
};

class AntiTransposeRewriter : public  ReWriter {
    void rewrite(Operator op, Graph graph) override;
};

class TransposeBeforeGEMMRewriter : public  ReWriter {
    void rewrite(Operator op, Graph graph) override;
};

class TransposeBeforeGEMMPattern : public Pattern {
public:
    std::string name() const override {
        return "TransposeBeforeGEMMPattern";
    }

    bool match(Operator op) override;
};

class Rules {
public:
    static Rules& getInstance() {
        static Rules rules;
        return rules;
    }

    bool register_rule(const Ref<Pattern>& pattern, const Ref<ReWriter>& reWriter) {
        register_rules.insert({pattern, reWriter});
        return true;
    }

    void optimize(Operator op, Graph graph) {
        for (auto& pair: register_rules) {
            if (pair.first->match(op)) {
                pair.second->rewrite(op, graph);
            }
        }
    }
private:
    std::multimap<Ref<Pattern>, Ref<ReWriter>, PointerCmp<Pattern>> register_rules;
};
}

#define REGISTER_RULE(pattern, rewriter)                                                        \
    static const bool _SELECT(_register_rule_, __COUNTER__) =                                   \
        Rules::getInstance().register_rule(make_ref<pattern>(), make_ref<rewriter>());

