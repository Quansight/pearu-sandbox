#include "fftpack.h"
typedef complex_double complex128_t;

extern void zfft(complex_double * inout, int n, int direction, int howmany, int normalize);
