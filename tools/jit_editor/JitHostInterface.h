#pragma once
#include <functional>
// Implemented only by the JIT Editor. Production plugins do not use this path.
struct JitHostInterface {
    virtual ~JitHostInterface() = default;
    std::function<void()> requestHostReconfigure;
    virtual bool commitHostConfiguration() = 0;
};
