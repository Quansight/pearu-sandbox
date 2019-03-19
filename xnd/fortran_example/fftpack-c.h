//#include "fftpack.h"


typedef struct {double r,i;} complex_double;
typedef struct {float r,i;} complex_float;

typedef complex_double complex128_t;

extern void zfft(complex_double * inout, int n, int direction, int howmany, int normalize);
