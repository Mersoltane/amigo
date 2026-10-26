#ifndef AMIGO_SCALAR_AD_H
#define AMIGO_SCALAR_AD_H

// Scalar parts of A2D that components use without the matrix operations and their tests
#include "ad/a2dobj.h"
#include "ad/a2dscalarops.h"
#include "ad/a2dstack.h"
#include "ad/a2dvartuple.h"
#include "amigo.h"

namespace amigo {

// Call fn in its own stack frame to release its temporaries on return
template <class Fn>
AMIGO_HOST_DEVICE AMIGO_NOINLINE auto call_noinline(const Fn& fn) {
  return fn();
}

}  // namespace amigo

#endif  // AMIGO_SCALAR_AD_H
