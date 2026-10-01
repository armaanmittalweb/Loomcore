// Loomcore installed-package consumer — see this directory's README.md.
//
// Deliberately does the minimum that actually exercises the packaging
// claim: include a Loomcore header found via the installed package's
// include directory, link against loomcore::core resolved via
// find_package(Loomcore), and construct a real loomcore::Runtime (which
// forces ONNX Runtime environment initialization — see Runtime's
// constructor — so a missing/misplaced onnxruntime.dll/.so at the
// install prefix would fail here, not silently pass). It intentionally
// never calls loadGraph()/run(): proving the *package* is consumable
// doesn't require the reference ONNX models to be present.
#include "loomcore/runtime.h"

#include <iostream>

int main() {
    loomcore::Runtime runtime;
    std::cout << "Loomcore consumed via find_package(Loomcore): Runtime constructed OK.\n";
    return 0;
}
