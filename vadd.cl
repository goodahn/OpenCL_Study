__kernel void vmul(
                const int             START,
                const int             SIZE,
                __global const float* A,
                __global const float* B,
                __global       float* C)
{
    int i = get_global_id(0);
    int j, k;
    
    if (i < SIZE && i >= START) {
        C[i] = A[i]*B[i];
    }
}
