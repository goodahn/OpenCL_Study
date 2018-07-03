__kernel void mmul(
                const int             START,
                const int             IMAGE_SIZE,
                const int             NUM_LABELS,
                const int             BATCH_SIZE,
                __global       float* W,
                __global const float* X,
                __global       float* Y)
{
    int i = get_global_id(0);
    int j, k;
    
    if (i < IMAGE_SIZE) {
        for (j=START; j<NUM_LABELS; j++) {
            for (k=0; k<BATCH_SIZE; k++) {
                Y[j*BATCH_SIZE+k] += W[j*IMAGE_SIZE+i] * X[i+BATCH_SIZE*k];
            }
        }
    }
}

__kernel void mmul2(
                const int             START,
                const int             IMAGE_SIZE,
                const int             NUM_LABELS,
                const int             BATCH_SIZE,
                __global       float* W,
                __global const float* X,
                __global       float* Y)
{
    int i = get_global_id(0);
    int j, k;
    
    if (i < IMAGE_SIZE && i >= START) {
        for (j=START; j<NUM_LABELS; j++) {
            for (k=0; k<BATCH_SIZE; k++) {
                Y[j*BATCH_SIZE+k] += W[j*IMAGE_SIZE+i] * X[i*IMAGE_SIZE+k];
            }
        }
    }
}

__kernel void madd(
                const int             START,
                const          int             SIZE,
                __global const float* B,
                __global       float* Y)
{
    int i = get_global_id(0);
    int j, k;
    
    if (i < SIZE && i >= START) {
        Y[i] += B[i];
    }
}

__kernel void msub(
                const int             START,
                const          int             SIZE,
                const          float           coef,
                __global const float* B,
                __global       float* Y)
{
    int i = get_global_id(0);
    int j, k;
    
    if (i < SIZE && i >= START) {
        Y[i] -= B[i];
    }
}

__kernel void mexp(
                const int             START,
                const unsigned int             SIZE,
                __global       float* Y,
                __global       float* SUM)
{
    int i = get_global_id(0);
    
    if (i < SIZE && i >= START) {
        Y[i] = exp(Y[i]);
        SUM[0] += Y[i];
    }
}

__kernel void mnormalize(
                const int             START,
                const unsigned int             SIZE,
                __global       float* Y,
                __global const float* SUM)
{
    int i = get_global_id(0);
    
    if (i < SIZE && i >= START) {
        Y[i] /= SUM[0];
    }
}

//
//__kernel void update_weight(
//                const unsigned int             ROWS,
//                const unsigned int             COLS,
//                __global const float* Y,
//                __global const float* Y_,
//                __global const float* X,
//                __global       float* W,
//                __global const float* step_size)
//{
//   float sum = 0.0f;
//   float tmp_Y[10];
//   float entropy = 0.0f;
//   int i = get_global_id(0);
//   int j, k;
//
//   if (i < COLS) {
//       for (j=0; j<ROWS; j++) {
//           W[j*ROWS+i] = W[j*ROWS+i] - step_size*((Y[i]-Y_[i])*X[i]);
//       }
//   }
//}
