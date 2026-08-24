/*
One round of message passing of a site-and-relation map as hand-written
CUDA kernels on ``mma.sync`` tensor-core tiles, the CUDA backend of
``discopy.neural.fused``.  ``discopy.neural.fused_cuda`` compiles this
file at runtime with NVRTC, once per geometry, with the geometry as macros:

    D, S, H, HU, C, Y         widths of a leg, the state, the cell's and the
                              unit's hidden layer, the clue and the answer
    LEGS, MEMBERS             legs of a cell, members of a unit
    TOTAL, NC, CW, NU, UW     the flat state, cells and units per row and
                              their block widths
    UOFF                      where the units' blocks start in a row
    ECHO_C, ECHO_A            whether the clue and the answer are echoed

Every kernel is a template on the precision P: TF32 runs the GEMMs on
``mma.sync.m16n8k8`` with inputs rounded to TF32 (``cvt.rna``) and float32
accumulation, as cuBLAS does under ``torch.set_float32_matmul_precision
("high")``; FP32 and FP64 run the same fragments through plain fused
multiply-adds, slowly, and are the proof of the layouts (``gradcheck``
runs FP64).  The activations of a cell or unit never leave registers.

Tiles.  A warp holds 16 rows of one cell or unit; a ``[16, 8w]`` tile is
``w`` fragments of four values per lane, lane ``(g = lane >> 2, t = lane
& 3)`` holding ``[g][2t], [g][2t + 1], [g + 8][2t], [g + 8][2t + 1]``.
This is the accumulator layout of ``mma.m16n8k8``; it is also the layout
the kernels feed its A operand in, because they permute the reduction
index of every 8-wide k-tile -- hardware slot ``t`` is column ``2t`` and
slot ``t + 4`` is column ``2t + 1`` -- and read the B operand, a weight
matrix ``W[n][k]`` staged in shared memory, with the same permutation:
``b0 = W[n0 + g][k0 + 2t]``, ``b1 = W[n0 + g][k0 + 2t + 1]``, one 8-byte
load.  So the output of one linear layer is the input of the next with
no data movement, elementwise operations are lane-local, and a row's
reduction is a sum over the four lanes of its quad.  Every width is
padded to a multiple of 8 with room for one more column: the input of a
linear layer carries a 1 in its first padded column and the staged
weights carry the bias there, so a bias costs nothing; the other padded
columns are zero on both operands.  Row strides in shared memory are 8
or 24 modulo 32 so that the eight rows of a fragment load hit distinct
banks.  Work is dealt to warps: warp ``w`` of the grid takes the 16-row
tiles ``w, w + warps, ...`` of the (rows, cell) items.
*/

#define FULL 0xffffffffu
enum { TF32 = 0, FP32 = 1, FP64 = 2 };

template <int P> struct Prec;
template <> struct Prec<TF32> {
    typedef float T; typedef float2 T2; typedef float4 T4; typedef unsigned R; typedef uint2 R2;
};
template <> struct Prec<FP32> {
    typedef float T; typedef float2 T2; typedef float4 T4; typedef float R; typedef float2 R2;
};
template <> struct Prec<FP64> {
    typedef double T; typedef double2 T2; typedef double4 T4; typedef double R; typedef double2 R2;
};

#define TILES(w) (((w) + 7) / 8)
#define PAD(w) (8 * TILES(w))
#define STRIDE(w) (((w) % 32 == 8 || (w) % 32 == 24) ? (w) : (w) + 8)

constexpr int X = H + C + Y;
constexpr int DT = TILES(D + 1), ST = TILES(S + 1), HT = TILES(H + 1), UT = TILES(HU + 1);
constexpr int CT = TILES(C), YT = TILES(Y), XT = TILES(C + Y + 1);
constexpr int DP = 8 * DT, SP = 8 * ST, HP = 8 * HT, UP = 8 * UT, XP = 8 * XT;
constexpr int O_SI = LEGS * D, O_SO = O_SI + S, O_CI = O_SO + S;
constexpr int O_CO = O_CI + C, O_AI = O_CO + C, O_AO = O_AI + Y;
constexpr int SV = 4 * SP + HP + 8;
constexpr int THREADS = 256, WARPS = THREADS / 32;

/* --- lanes, rounding, activations ------------------------------------ */

__device__ __forceinline__ int lane_g() { return (threadIdx.x & 31) >> 2; }
__device__ __forceinline__ int lane_t() { return threadIdx.x & 3; }

template <int P> __device__ __forceinline__ typename Prec<P>::R rnd(typename Prec<P>::T x) {
    return x;
}
template <> __device__ __forceinline__ unsigned rnd<TF32>(float x) {
    unsigned u = __float_as_uint(x);
    asm("cvt.rna.tf32.f32 %0, %0;" : "+r"(u));
    return u;
}

__device__ __forceinline__ float ex(float x) { return expf(x); }
__device__ __forceinline__ double ex(double x) { return exp(x); }
__device__ __forceinline__ float th(float x) { return tanhf(x); }
__device__ __forceinline__ double th(double x) { return tanh(x); }
__device__ __forceinline__ float sq(float x) { return sqrtf(x); }
__device__ __forceinline__ double sq(double x) { return sqrt(x); }
template <typename T> __device__ __forceinline__ T sigmoid(T x) { return (T) 1 / ((T) 1 + ex(-x)); }
template <typename T> __device__ __forceinline__ T relu(T x) { return x > (T) 0 ? x : (T) 0; }

template <typename T> __device__ __forceinline__ T quad_sum(T x) {
    x += __shfl_xor_sync(FULL, x, 1);
    x += __shfl_xor_sync(FULL, x, 2);
    return x;
}

/* --- the tensor-core tile --------------------------------------------- */

/* c += a @ b on one [16, 8] x [8, 8] tile through plain multiply-adds, the
   fragments gathered with shuffles: one out-of-line copy, so that the FP32
   and FP64 kernels compile in seconds. */
template <int P> __device__ __noinline__ void mma_emulated(
        typename Prec<P>::T c[4], const typename Prec<P>::R a[4], const typename Prec<P>::R b[2]) {
    typedef typename Prec<P>::T T;
    const int g = lane_g(), t = lane_t();
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        T ag = __shfl_sync(FULL, a[k & 1], g * 4 + (k >> 1));
        T ag8 = __shfl_sync(FULL, a[2 + (k & 1)], g * 4 + (k >> 1));
        T bn0 = __shfl_sync(FULL, b[k & 1], (2 * t) * 4 + (k >> 1));
        T bn1 = __shfl_sync(FULL, b[k & 1], (2 * t + 1) * 4 + (k >> 1));
        c[0] += ag * bn0; c[1] += ag * bn1; c[2] += ag8 * bn0; c[3] += ag8 * bn1;
    }
}

/* c += a @ b on one [16, 8] x [8, 8] tile, fragments as described above. */
template <int P> __device__ __forceinline__ void mma(
        typename Prec<P>::T c[4], const typename Prec<P>::R a[4], const typename Prec<P>::R b[2]) {
    if constexpr (P == TF32) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        mma_emulated<P>(c, a, b);
    }
}

/* The B fragment of a staged weight matrix: rows n0 + g, columns k0 + 2t, k0 + 2t + 1. */
template <int P> __device__ __forceinline__ void frag_b(
        typename Prec<P>::R b[2], const typename Prec<P>::R* w, int stride, int n0, int k0) {
    typename Prec<P>::R2 v = *reinterpret_cast<const typename Prec<P>::R2*>(
        w + (n0 + lane_g()) * stride + k0 + 2 * lane_t());
    b[0] = v.x; b[1] = v.y;
}

/* c[i] += a @ w[8i.., col0 + 8j..] over NT output tiles and KT input tiles:
   per input tile, every B fragment is loaded before the NT independent
   mma's, so that the loads pipeline and the mma's never wait on one another. */
template <int P, int NT, int KT> __device__ __forceinline__ void gemm(
        typename Prec<P>::T c[NT][4], const typename Prec<P>::R a[KT][4],
        const typename Prec<P>::R* w, int stride, int col0) {
    #pragma unroll
    for (int j = 0; j < KT; ++j) {
        typename Prec<P>::R b[NT][2];
        #pragma unroll
        for (int i = 0; i < NT; ++i) frag_b<P>(b[i], w, stride, 8 * i, col0 + 8 * j);
        #pragma unroll
        for (int i = 0; i < NT; ++i) mma<P>(c[i], a[j], b[i]);
    }
}

/* A bias as column K of a staged [NP, .] matrix, rounded, zero past N. */
template <int P, int NP, int STR, int N> __device__ void stage_bias(
        typename Prec<P>::R* dst, const typename Prec<P>::T* b) {
    for (int r = threadIdx.x; r < NP; r += blockDim.x)
        dst[r * STR] = rnd<P>(r < N ? __ldg(b + r) : (typename Prec<P>::T) 0);
}

/* Stage W[:N, :K] (row stride ldw) rounded and zero-padded to [NP, KP] in dst of row stride STR. */
template <int P, int NP, int STR, int N, int K, int KP> __device__ void stage(
        typename Prec<P>::R* dst, const typename Prec<P>::T* w, int ldw) {
    typedef typename Prec<P>::T T;
    constexpr int KC = (KP + 31) / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll 4
    for (int r = warp; r < NP; r += THREADS / 32) {
        T v[KC];
        #pragma unroll
        for (int q = 0; q < KC; ++q) {
            int c = lane + 32 * q;
            v[q] = (r < N && c < K) ? __ldg(w + r * ldw + c) : (T) 0;
        }
        #pragma unroll
        for (int q = 0; q < KC; ++q) {
            int c = lane + 32 * q;
            if (c < KP) dst[r * STR + c] = rnd<P>(v[q]);
        }
    }
}

/* --- tiles in registers ----------------------------------------------- */

template <typename T, int N> __device__ __forceinline__ void zero(T v[N][4]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) v[i][0] = v[i][1] = v[i][2] = v[i][3] = (T) 0;
}

template <typename T, int N> __device__ __forceinline__ void copy(T dst[N][4], const T src[N][4]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) { dst[i][0] = src[i][0]; dst[i][1] = src[i][1]; dst[i][2] = src[i][2]; dst[i][3] = src[i][3]; }
}

/* v += b[column] on every row, zero past width. */
template <typename T, int N> __device__ __forceinline__ void add_bias(T v[N][4], const T* b, int width) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        T b0 = c < width ? b[c] : (T) 0, b1 = c + 1 < width ? b[c + 1] : (T) 0;
        v[i][0] += b0; v[i][1] += b1; v[i][2] += b0; v[i][3] += b1;
    }
}

template <int P, int N> __device__ __forceinline__ void round_tiles(
        typename Prec<P>::R a[N][4], const typename Prec<P>::T v[N][4]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) { a[i][0] = rnd<P>(v[i][0]); a[i][1] = rnd<P>(v[i][1]); a[i][2] = rnd<P>(v[i][2]); a[i][3] = rnd<P>(v[i][3]); }
}

template <int P, int N> __device__ __forceinline__ void relu_round(
        typename Prec<P>::R a[N][4], typename Prec<P>::T v[N][4]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) { v[i][q] = relu(v[i][q]); a[i][q] = rnd<P>(v[i][q]); }
    }
}

/* The tiles of columns [0, width) of the two rows at p0, p1 of a flat matrix. */
template <typename T, int N> __device__ __forceinline__ void load_tiles(
        T v[N][4], const T* p0, const T* p1, int width, bool v0, bool v1) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        v[i][0] = (v0 && c < width) ? p0[c] : (T) 0;
        v[i][1] = (v0 && c + 1 < width) ? p0[c + 1] : (T) 0;
        v[i][2] = (v1 && c < width) ? p1[c] : (T) 0;
        v[i][3] = (v1 && c + 1 < width) ? p1[c + 1] : (T) 0;
    }
}

/* The 1 of the bias column: column ``width`` of both rows. */
template <int P, int N> __device__ __forceinline__ void set_one(typename Prec<P>::R a[N][4], int width) {
    const int i = width / 8, q = (width % 8) & 1;
    if (lane_t() == (width % 8) / 2) { a[i][q] = rnd<P>((typename Prec<P>::T) 1); a[i][2 + q] = a[i][q]; }
}

/* The A operand of a linear layer: rounded tiles with the bias column's 1. */
template <int P, int N> __device__ __forceinline__ void load_a(
        typename Prec<P>::R a[N][4], const typename Prec<P>::T* p0, const typename Prec<P>::T* p1,
        int width, bool v0, bool v1) {
    typename Prec<P>::T v[N][4];
    load_tiles<typename Prec<P>::T, N>(v, p0, p1, width, v0, v1);
    round_tiles<P, N>(a, v);
    set_one<P, N>(a, width);
}

/* The warp's 16-row tiles of the items, warp ``w`` of the grid taking every WARPS * gridDim.x-th. */
__device__ __forceinline__ int first_item() { return blockIdx.x * WARPS + (threadIdx.x >> 5); }
__device__ __forceinline__ int item_step() { return gridDim.x * WARPS; }

/* The tiles of cat(clue, answer), read from their two blocks of a cell. */
template <typename T> __device__ __forceinline__ T given(const T* p, int c) {
    return c < C ? p[O_CI + c] : c < C + Y ? p[O_AI + c - C] : (T) 0;
}
template <int P> __device__ __forceinline__ void load_given(
        typename Prec<P>::R a[XT][4], const typename Prec<P>::T* p0, const typename Prec<P>::T* p1,
        bool v0, bool v1) {
    typedef typename Prec<P>::T T;
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < XT; ++i) {
        int c = 8 * i + 2 * t;
        a[i][0] = rnd<P>(v0 ? given(p0, c) : (T) 0);
        a[i][1] = rnd<P>(v0 ? given(p0, c + 1) : (T) 0);
        a[i][2] = rnd<P>(v1 ? given(p1, c) : (T) 0);
        a[i][3] = rnd<P>(v1 ? given(p1, c + 1) : (T) 0);
    }
}

template <typename T, int N> __device__ __forceinline__ void store_tiles(
        T* q0, T* q1, int width, const T v[N][4], bool v0, bool v1) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        if (v0 && c < width) q0[c] = v[i][0];
        if (v0 && c + 1 < width) q0[c + 1] = v[i][1];
        if (v1 && c < width) q1[c] = v[i][2];
        if (v1 && c + 1 < width) q1[c + 1] = v[i][3];
    }
}

/* store_tiles on an 8-byte aligned buffer, two columns per store. */
template <int P, int N> __device__ __forceinline__ void store_tiles2(
        typename Prec<P>::T* q0, typename Prec<P>::T* q1, int width, const typename Prec<P>::T v[N][4],
        bool v0, bool v1) {
    typedef typename Prec<P>::T2 T2;
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        if (c + 1 < width) {
            if (v0) *reinterpret_cast<T2*>(q0 + c) = T2{v[i][0], v[i][1]};
            if (v1) *reinterpret_cast<T2*>(q1 + c) = T2{v[i][2], v[i][3]};
        } else if (c < width) {
            if (v0) q0[c] = v[i][0];
            if (v1) q1[c] = v[i][2];
        }
    }
}

/* load_tiles on an 8-byte aligned buffer, two columns per load. */
template <int P, int N> __device__ __forceinline__ void load_tiles2(
        typename Prec<P>::T v[N][4], const typename Prec<P>::T* p0, const typename Prec<P>::T* p1,
        int width, bool v0, bool v1) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::T2 T2;
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        v[i][0] = v[i][1] = v[i][2] = v[i][3] = (T) 0;
        if (c + 1 < width) {
            if (v0) { T2 u = *reinterpret_cast<const T2*>(p0 + c); v[i][0] = u.x; v[i][1] = u.y; }
            if (v1) { T2 u = *reinterpret_cast<const T2*>(p1 + c); v[i][2] = u.x; v[i][3] = u.y; }
        } else if (c < width) {
            if (v0) v[i][0] = p0[c];
            if (v1) v[i][2] = p1[c];
        }
    }
}

/* Store, adding the injected messages at the same positions when there are any. */
template <typename T, int N> __device__ __forceinline__ void emit_tiles(
        T* out, const T* init, long o0, long o1, int width, const T v[N][4], bool v0, bool v1, bool has_init) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        if (v0 && c < width) out[o0 + c] = v[i][0] + (has_init ? init[o0 + c] : (T) 0);
        if (v0 && c + 1 < width) out[o0 + c + 1] = v[i][1] + (has_init ? init[o0 + c + 1] : (T) 0);
        if (v1 && c < width) out[o1 + c] = v[i][2] + (has_init ? init[o1 + c] : (T) 0);
        if (v1 && c + 1 < width) out[o1 + c + 1] = v[i][3] + (has_init ? init[o1 + c + 1] : (T) 0);
    }
}

/* The sums of the two rows of a tile set over columns [0, width). */
template <typename T, int N> __device__ __forceinline__ void row_sums(T& s0, T& s1, const T v[N][4], int width) {
    const int t = lane_t();
    s0 = s1 = (T) 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int c = 8 * i + 2 * t;
        if (c < width) { s0 += v[i][0]; s1 += v[i][2]; }
        if (c + 1 < width) { s0 += v[i][1]; s1 += v[i][3]; }
    }
    s0 = quad_sum(s0);
    s1 = quad_sum(s1);
}

/* --- the cell ----------------------------------------------------------- */

constexpr int STR_W1 = STRIDE(SP + DP), STR_W2 = STRIDE(HP), STR_IH = STRIDE(HP + XP);
constexpr int STR_HH = STRIDE(SP), STR_E = STRIDE(SP);
constexpr int SM_CELL_FWD = HP * STR_W1 + HP * STR_W2 + 3 * SP * STR_IH + 3 * SP * STR_HH + DP * STR_E;

/*
cell_fwd: Site.forward, a warp per 16 rows of one cell, routed on the way
out.  A persistent grid: each block stages the weights once and its warps
loop over the (16 rows, cell) items.  Reads the
cell's block of ``x`` -- LEGS legs, state, clue, answer -- and writes the
belief to the LEGS positions ``pinv`` routes the legs to, the new state
to both state ports, the echoes to the clue and answer ports, adding
``init`` at every position when ``has_init``.  When ``do_save``, keeps
``[r, z, n, hn | pool | mu, rstd]`` per (row, cell) in ``save``.
*/
template <int P> __global__ void __launch_bounds__(THREADS, P == TF32 ? 2 : 1) cell_fwd(
        const typename Prec<P>::T* x, typename Prec<P>::T* out, const typename Prec<P>::T* init,
        typename Prec<P>::T* save, const long long* pinv, int rows, double eps, int has_init, int do_save,
        const typename Prec<P>::T* w1, const typename Prec<P>::T* b1,
        const typename Prec<P>::T* w2, const typename Prec<P>::T* b2,
        const typename Prec<P>::T* wih, const typename Prec<P>::T* whh,
        const typename Prec<P>::T* bih, const typename Prec<P>::T* bhh,
        const typename Prec<P>::T* gamma, const typename Prec<P>::T* beta,
        const typename Prec<P>::T* we, const typename Prec<P>::T* be) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::R R;
    extern __shared__ __align__(16) unsigned char smem[];
    R* w1s = (R*) smem;
    R* w2s = w1s + HP * STR_W1;
    R* wihs = w2s + HP * STR_W2;
    R* whhs = wihs + 3 * SP * STR_IH;
    R* wes = whhs + 3 * SP * STR_HH;
    stage<P, HP, STR_W1, H, S, SP>(w1s, w1, S + D);
    stage<P, HP, STR_W1, H, D, DP>(w1s + SP, w1 + S, S + D);
    stage<P, HP, STR_W2, H, H, HP>(w2s, w2, H);
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
        stage<P, SP, STR_IH, S, H, HP>(wihs + q * SP * STR_IH, wih + q * S * X, X);
        stage<P, SP, STR_IH, S, C + Y, XP>(wihs + q * SP * STR_IH + HP, wih + q * S * X + H, X);
        stage<P, SP, STR_HH, S, S, SP>(whhs + q * SP * STR_HH, whh + q * S * S, S);
    }
    stage<P, DP, STR_E, D, S, SP>(wes, we, S);
    __syncthreads();
    stage_bias<P, HP, STR_W1, H>(w1s + S, b1);
    stage_bias<P, HP, STR_W2, H>(w2s + H, b2);
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
        stage_bias<P, SP, STR_IH, S>(wihs + q * SP * STR_IH + H, bih + q * S);
        stage_bias<P, SP, STR_HH, S>(whhs + q * SP * STR_HH + S, bhh + q * S);
    }
    stage_bias<P, DP, STR_E, D>(wes + S, be);
    __syncthreads();

    const int g = lane_g(), t = lane_t(), items = ((rows + 15) / 16) * NC;
    #pragma unroll 1
    for (int item = first_item(); item < items; item += item_step()) {
    const int cell = item % NC;
    const int r0 = (item / NC) * 16 + g, r1 = r0 + 8;
    const bool v0 = r0 < rows, v1 = r1 < rows;
    const long base0 = (long) r0 * TOTAL + cell * CW, base1 = (long) r1 * TOTAL + cell * CW;
    const T* p0 = x + base0;
    const T* p1 = x + base1;

    /* the encoder, mean-pooled over the legs */
    R xa[ST + DT][4];
    load_a<P, ST>(xa, p0 + O_SI, p1 + O_SI, S, v0, v1);
    T pool[HT][4];
    zero<T, HT>(pool);
    T leg[DT][4];
    load_tiles<T, DT>(leg, p0, p1, D, v0, v1);
    #pragma unroll 1
    for (int i = 0; i < LEGS; ++i) {
        round_tiles<P, DT>(xa + ST, leg);
        if (i + 1 < LEGS) load_tiles<T, DT>(leg, p0 + (i + 1) * D, p1 + (i + 1) * D, D, v0, v1);
        T h[HT][4];
        zero<T, HT>(h);
        gemm<P, HT, ST + DT>(h, xa, w1s, STR_W1, 0);
        R ha[HT][4];
        relu_round<P, HT>(ha, h);
        set_one<P, HT>(ha, H);
        gemm<P, HT, HT>(pool, ha, w2s, STR_W2, 0);
    }
    #pragma unroll
    for (int i = 0; i < HT; ++i) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) pool[i][q] /= (T) LEGS;
    }
    const long sv0 = ((long) r0 * NC + cell) * SV, sv1 = ((long) r1 * NC + cell) * SV;
    if (do_save) store_tiles2<P, HT>(save + sv0 + 4 * SP, save + sv1 + 4 * SP, H, pool, v0, v1);

    /* the GRU: the input halves of the three gates, then the hidden halves */
    T acc[3][ST][4];
    {
        R ga[HT + XT][4];
        round_tiles<P, HT>(ga, pool);
        set_one<P, HT>(ga, H);
        load_given<P>(ga + HT, p0, p1, v0, v1);
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            zero<T, ST>(acc[q]);
            gemm<P, ST, HT + XT>(acc[q], ga, wihs + q * SP * STR_IH, STR_IH, 0);
        }
    }
    T hn[ST][4];
    zero<T, ST>(hn);
    {
        R sa[ST][4];
        load_a<P, ST>(sa, p0 + O_SI, p1 + O_SI, S, v0, v1);
        gemm<P, ST, ST>(acc[0], sa, whhs, STR_HH, 0);
        gemm<P, ST, ST>(acc[1], sa, whhs + SP * STR_HH, STR_HH, 0);
        gemm<P, ST, ST>(hn, sa, whhs + 2 * SP * STR_HH, STR_HH, 0);
    }
    T h[ST][4];
    {
        T s[ST][4];
        load_tiles<T, ST>(s, p0 + O_SI, p1 + O_SI, S, v0, v1);
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                T r = sigmoid(acc[0][i][q]), z = sigmoid(acc[1][i][q]);
                T n = th(acc[2][i][q] + r * hn[i][q]);
                acc[0][i][q] = r; acc[1][i][q] = z; acc[2][i][q] = n;
                h[i][q] = ((T) 1 - z) * n + z * s[i][q];
            }
        }
    }
    if (do_save) {
        store_tiles2<P, ST>(save + sv0, save + sv1, S, acc[0], v0, v1);
        store_tiles2<P, ST>(save + sv0 + SP, save + sv1 + SP, S, acc[1], v0, v1);
        store_tiles2<P, ST>(save + sv0 + 2 * SP, save + sv1 + 2 * SP, S, acc[2], v0, v1);
        store_tiles2<P, ST>(save + sv0 + 3 * SP, save + sv1 + 3 * SP, S, hn, v0, v1);
    }

    /* the layer norm and the emission */
    T mu0, mu1, var0, var1;
    row_sums<T, ST>(mu0, mu1, h, S);
    mu0 /= (T) S; mu1 /= (T) S;
    #pragma unroll
    for (int i = 0; i < ST; ++i) {
        int c = 8 * i + 2 * t;
        h[i][0] = c < S ? h[i][0] - mu0 : (T) 0;
        h[i][1] = c + 1 < S ? h[i][1] - mu0 : (T) 0;
        h[i][2] = c < S ? h[i][2] - mu1 : (T) 0;
        h[i][3] = c + 1 < S ? h[i][3] - mu1 : (T) 0;
        hn[i][0] = h[i][0] * h[i][0]; hn[i][1] = h[i][1] * h[i][1];
        hn[i][2] = h[i][2] * h[i][2]; hn[i][3] = h[i][3] * h[i][3];
    }
    row_sums<T, ST>(var0, var1, hn, S);
    const T rstd0 = (T) 1 / sq(var0 / (T) S + (T) eps), rstd1 = (T) 1 / sq(var1 / (T) S + (T) eps);
    if (do_save && t == 0) {
        if (v0) { save[sv0 + 4 * SP + HP] = mu0; save[sv0 + 4 * SP + HP + 1] = rstd0; }
        if (v1) { save[sv1 + 4 * SP + HP] = mu1; save[sv1 + 4 * SP + HP + 1] = rstd1; }
    }
    #pragma unroll
    for (int i = 0; i < ST; ++i) {
        int c = 8 * i + 2 * t;
        T g0 = c < S ? gamma[c] : (T) 0, g1 = c + 1 < S ? gamma[c + 1] : (T) 0;
        T e0 = c < S ? beta[c] : (T) 0, e1 = c + 1 < S ? beta[c + 1] : (T) 0;
        h[i][0] = h[i][0] * rstd0 * g0 + e0; h[i][1] = h[i][1] * rstd0 * g1 + e1;
        h[i][2] = h[i][2] * rstd1 * g0 + e0; h[i][3] = h[i][3] * rstd1 * g1 + e1;
    }
    T belief[DT][4];
    zero<T, DT>(belief);
    {
        R ya[ST][4];
        round_tiles<P, ST>(ya, h);
        set_one<P, ST>(ya, S);
        gemm<P, DT, ST>(belief, ya, wes, STR_E, 0);
    }

    /* the routed stores */
    #pragma unroll 1
    for (int i = 0; i < LEGS; ++i) {
        const long dest = pinv[cell * CW + i * D];
        emit_tiles<T, DT>(out, init, (long) r0 * TOTAL + dest, (long) r1 * TOTAL + dest, D, belief, v0, v1, has_init);
    }
    emit_tiles<T, ST>(out, init, base0 + O_SI, base1 + O_SI, S, h, v0, v1, has_init);
    emit_tiles<T, ST>(out, init, base0 + O_SO, base1 + O_SO, S, h, v0, v1, has_init);
    {
        T c[CT][4];
        if (ECHO_C) load_tiles<T, CT>(c, p0 + O_CI, p1 + O_CI, C, v0, v1); else zero<T, CT>(c);
        emit_tiles<T, CT>(out, init, base0 + O_CI, base1 + O_CI, C, c, v0, v1, has_init);
        emit_tiles<T, CT>(out, init, base0 + O_CO, base1 + O_CO, C, c, v0, v1, has_init);
    }
    {
        T a[YT][4];
        if (ECHO_A) load_tiles<T, YT>(a, p0 + O_AI, p1 + O_AI, Y, v0, v1); else zero<T, YT>(a);
        emit_tiles<T, YT>(out, init, base0 + O_AI, base1 + O_AI, Y, a, v0, v1, has_init);
        emit_tiles<T, YT>(out, init, base0 + O_AO, base1 + O_AO, Y, a, v0, v1, has_init);
    }
    }
}

/* --- the unit ----------------------------------------------------------- */

constexpr int STR_PHI = STRIDE(DP), STR_R1 = STRIDE(DP + UP), STR_R2 = STRIDE(UP);
constexpr int SM_UNIT_FWD = UP * STR_PHI + UP * STR_R1 + DP * STR_R2;

/*
unit_fwd: Relation.forward, a warp per 16 rows of one unit, routed on the
way out, on a persistent grid whose warps loop over the (16 rows, unit)
items.  Reads the
unit's MEMBERS legs of ``x`` and writes each member's answer to the
position ``pinv`` routes it to, adding ``init`` there when ``has_init``.
Nothing is saved: the backward recomputes the unit from its legs.
*/
template <int P> __global__ void __launch_bounds__(THREADS, P == TF32 ? 2 : 1) unit_fwd(
        const typename Prec<P>::T* x, typename Prec<P>::T* out, const typename Prec<P>::T* init,
        const long long* pinv, int rows, int has_init,
        const typename Prec<P>::T* wphi, const typename Prec<P>::T* bphi,
        const typename Prec<P>::T* wr1, const typename Prec<P>::T* br1,
        const typename Prec<P>::T* wr2, const typename Prec<P>::T* br2) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::R R;
    extern __shared__ __align__(16) unsigned char smem[];
    R* wphis = (R*) smem;
    R* wr1s = wphis + UP * STR_PHI;
    R* wr2s = wr1s + UP * STR_R1;
    stage<P, UP, STR_PHI, HU, D, DP>(wphis, wphi, D);
    stage<P, UP, STR_R1, HU, D, DP>(wr1s, wr1, D + HU);
    stage<P, UP, STR_R1, HU, HU, UP>(wr1s + DP, wr1 + D, D + HU);
    stage<P, DP, STR_R2, D, HU, UP>(wr2s, wr2, HU);
    __syncthreads();
    stage_bias<P, UP, STR_PHI, HU>(wphis + D, bphi);
    stage_bias<P, UP, STR_R1, HU>(wr1s + D, br1);
    stage_bias<P, DP, STR_R2, D>(wr2s + HU, br2);
    __syncthreads();

    const int g = lane_g(), items = ((rows + 15) / 16) * NU;
    #pragma unroll 1
    for (int item = first_item(); item < items; item += item_step()) {
    const int unit = item % NU;
    const int r0 = (item / NU) * 16 + g, r1 = r0 + 8;
    const bool v0 = r0 < rows, v1 = r1 < rows;
    const T* p0 = x + (long) r0 * TOTAL + UOFF + unit * UW;
    const T* p1 = x + (long) r1 * TOTAL + UOFF + unit * UW;

    T zp[UT][4];
    {
        T pool[UT][4];
        zero<T, UT>(pool);
        T leg[DT][4];
        load_tiles<T, DT>(leg, p0, p1, D, v0, v1);
        #pragma unroll 1
        for (int i = 0; i < MEMBERS; ++i) {
            R la[DT][4];
            round_tiles<P, DT>(la, leg);
            set_one<P, DT>(la, D);
            if (i + 1 < MEMBERS) load_tiles<T, DT>(leg, p0 + (i + 1) * D, p1 + (i + 1) * D, D, v0, v1);
            T h[UT][4];
            zero<T, UT>(h);
            gemm<P, UT, DT>(h, la, wphis, STR_PHI, 0);
            #pragma unroll
            for (int j = 0; j < UT; ++j) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) pool[j][q] += relu(h[j][q]);
            }
        }
        R pa[UT][4];
        round_tiles<P, UT>(pa, pool);
        zero<T, UT>(zp);
        gemm<P, UT, UT>(zp, pa, wr1s, STR_R1, DP);
    }
    T leg[DT][4];
    load_tiles<T, DT>(leg, p0, p1, D, v0, v1);
    #pragma unroll 1
    for (int i = 0; i < MEMBERS; ++i) {
        R la[DT][4];
        round_tiles<P, DT>(la, leg);
        set_one<P, DT>(la, D);
        if (i + 1 < MEMBERS) load_tiles<T, DT>(leg, p0 + (i + 1) * D, p1 + (i + 1) * D, D, v0, v1);
        T h[UT][4];
        copy<T, UT>(h, zp);
        gemm<P, UT, DT>(h, la, wr1s, STR_R1, 0);
        R ra[UT][4];
        relu_round<P, UT>(ra, h);
        set_one<P, UT>(ra, HU);
        T o[DT][4];
        zero<T, DT>(o);
        gemm<P, DT, UT>(o, ra, wr2s, STR_R2, 0);
        const long dest = pinv[UOFF + unit * UW + i * D];
        emit_tiles<T, DT>(out, init, (long) r0 * TOTAL + dest, (long) r1 * TOTAL + dest, D, o, v0, v1, has_init);
    }
    }
}

/* --- the backward ------------------------------------------------------- */

/* Stage W transposed: dst[r][c] = W[c][r] for r < RN, c < CN, zero-padded to [RP, CP] at row stride STR. */
template <int P, int RP, int STR, int RN, int CN, int CP> __device__ void stage_t(
        typename Prec<P>::R* dst, const typename Prec<P>::T* w, int ldw) {
    typedef typename Prec<P>::T T;
    constexpr int KC = (CP + 31) / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll 4
    for (int r = warp; r < RP; r += THREADS / 32) {
        T v[KC];
        #pragma unroll
        for (int q = 0; q < KC; ++q) {
            int c = lane + 32 * q;
            v[q] = (r < RN && c < CN) ? __ldg(w + c * ldw + r) : (T) 0;
        }
        #pragma unroll
        for (int q = 0; q < KC; ++q) {
            int c = lane + 32 * q;
            if (c < CP) dst[r * STR + c] = rnd<P>(v[q]);
        }
    }
}

/* A warp's tiles parked in its own shared-memory scratch, four values per lane per tile. */
template <int P, int N> __device__ __forceinline__ void stash(
        typename Prec<P>::T* s, const typename Prec<P>::T v[N][4]) {
    typedef typename Prec<P>::T4 T4;
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < N; ++i)
        *reinterpret_cast<T4*>(s + (i * 32 + lane) * 4) = T4{v[i][0], v[i][1], v[i][2], v[i][3]};
}
template <int P> __device__ __forceinline__ void fetch(
        const typename Prec<P>::T* s, int i, typename Prec<P>::T v[4]) {
    typedef typename Prec<P>::T4 T4;
    T4 u = *reinterpret_cast<const T4*>(s + (i * 32 + (threadIdx.x & 31)) * 4);
    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
}

template <typename T> __device__ __forceinline__ void store_tile(
        T* q0, T* q1, int col0, int width, const T v[4], bool v0, bool v1) {
    int c = col0 + 2 * lane_t();
    if (v0 && c < width) q0[c] = v[0];
    if (v0 && c + 1 < width) q0[c + 1] = v[1];
    if (v1 && c < width) q1[c] = v[2];
    if (v1 && c + 1 < width) q1[c + 1] = v[3];
}

template <typename T> __device__ __forceinline__ void store_one(T* q0, T* q1, T value, bool v0, bool v1) {
    if (lane_t() == 0) {
        if (v0) *q0 = value;
        if (v1) *q1 = value;
    }
}

/* The tiles of cat(clue, answer) read from the blocks at oc and oa of a cell. */
template <typename T> __device__ __forceinline__ T given_at(const T* p, int c, int oc, int oa) {
    return c < C ? p[oc + c] : c < C + Y ? p[oa + c - C] : (T) 0;
}
template <typename T> __device__ __forceinline__ void load_given_tiles(
        T v[XT][4], const T* p0, const T* p1, int oc, int oa, bool v0, bool v1) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < XT; ++i) {
        int c = 8 * i + 2 * t;
        v[i][0] = v0 ? given_at(p0, c, oc, oa) : (T) 0;
        v[i][1] = v0 ? given_at(p0, c + 1, oc, oa) : (T) 0;
        v[i][2] = v1 ? given_at(p1, c, oc, oa) : (T) 0;
        v[i][3] = v1 ? given_at(p1, c + 1, oc, oa) : (T) 0;
    }
}
template <typename T> __device__ __forceinline__ void put_given(T* p, int c, int oc, int oa, T value) {
    if (c < C) p[oc + c] = value; else if (c < C + Y) p[oa + c - C] = value;
}
template <typename T> __device__ __forceinline__ void store_given_tiles(
        T* q0, T* q1, int oc, int oa, const T v[XT][4], bool v0, bool v1) {
    const int t = lane_t();
    #pragma unroll
    for (int i = 0; i < XT; ++i) {
        int c = 8 * i + 2 * t;
        if (v0) { put_given(q0, c, oc, oa, v[i][0]); put_given(q0, c + 1, oc, oa, v[i][1]); }
        if (v1) { put_given(q1, c, oc, oa, v[i][2]); put_given(q1, c + 1, oc, oa, v[i][3]); }
    }
}

template <typename T, int N> __device__ __forceinline__ void add_tiles(T v[N][4], const T w[N][4]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) { v[i][0] += w[i][0]; v[i][1] += w[i][1]; v[i][2] += w[i][2]; v[i][3] += w[i][3]; }
}

constexpr int STR_HHT = STRIDE(3 * SP), STR_IHT = STRIDE(3 * SP), STR_ET = STRIDE(DP);
constexpr int SM_CELL_BWD_GATE = SP * STR_HHT + (HP + XP) * STR_IHT + SP * STR_ET;
constexpr int K_X = 4 * S, K_S = K_X + X + 1, K_Y = K_S + S + 1, K_B = K_Y + S + 1, K_G = K_B + D;
constexpr int KW_GATE = K_G + 2 * S;

/*
cell_bwd_gate: the first half of the backward of cell_fwd, a warp per
16 rows of one cell, from the output gradient ``g`` down to the pooled
encoding, on a persistent grid.  Writes the gradients of the state, clue
and answer ports to ``gin``, the pooled gradient per (row, cell) to
``pool`` for cell_bwd_encode, and per (row, cell) the row of ``keep``
that fused._cell_grads reads: ``[g_ar, g_az, g_hn, g_an | pool, c, a, 1 |
s, 1 | y, 1 | g_belief | g_y * xhat, g_y]``.
*/
template <int P> __global__ void __launch_bounds__(THREADS, P == TF32 ? 2 : 1) cell_bwd_gate(
        const typename Prec<P>::T* x, const typename Prec<P>::T* g, typename Prec<P>::T* gin,
        const typename Prec<P>::T* save, typename Prec<P>::T* pool, typename Prec<P>::T* keep,
        const long long* pinv, int rows,
        const typename Prec<P>::T* wih, const typename Prec<P>::T* whh,
        const typename Prec<P>::T* gamma, const typename Prec<P>::T* beta, const typename Prec<P>::T* we) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::R R;
    extern __shared__ __align__(16) unsigned char smem[];
    R* whhT = (R*) smem;
    R* wihT = whhT + SP * STR_HHT;
    R* weT = wihT + (HP + XP) * STR_IHT;
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
        stage_t<P, SP, STR_HHT, S, S, SP>(whhT + q * SP, whh + q * S * S, S);
        stage_t<P, HP, STR_IHT, H, S, SP>(wihT + q * SP, wih + q * S * X, X);
        stage_t<P, XP, STR_IHT, C + Y, S, SP>(wihT + HP * STR_IHT + q * SP, wih + q * S * X + H, X);
    }
    stage_t<P, SP, STR_ET, S, D, DP>(weT, we, S);
    __syncthreads();

    const int gl = lane_g(), t = lane_t(), items = ((rows + 15) / 16) * NC;
    #pragma unroll 1
    for (int item = first_item(); item < items; item += item_step()) {
    const int cell = item % NC;
    const int r0 = (item / NC) * 16 + gl, r1 = r0 + 8;
    const bool v0 = r0 < rows, v1 = r1 < rows;
    const long base0 = (long) r0 * TOTAL + cell * CW, base1 = (long) r1 * TOTAL + cell * CW;
    const long m0 = (long) r0 * NC + cell, m1 = (long) r1 * NC + cell;
    const long sv0 = m0 * SV, sv1 = m1 * SV;
    T* k0 = keep + m0 * KW_GATE;
    T* k1 = keep + m1 * KW_GATE;
    const T* p0 = x + base0;
    const T* p1 = x + base1;

    /* what the weight gradients need: the GRU's input and the state, each with a one */
    {
        T pl[HT][4];
        load_tiles2<P, HT>(pl, save + sv0 + 4 * SP, save + sv1 + 4 * SP, H, v0, v1);
        store_tiles<T, HT>(k0 + K_X, k1 + K_X, H, pl, v0, v1);
        T ca[XT][4];
        load_given_tiles<T>(ca, p0, p1, O_CI, O_AI, v0, v1);
        store_tiles<T, XT>(k0 + K_X + H, k1 + K_X + H, C + Y, ca, v0, v1);
        store_one<T>(k0 + K_X + X, k1 + K_X + X, (T) 1, v0, v1);
    }
    T s[ST][4];
    load_tiles<T, ST>(s, p0 + O_SI, p1 + O_SI, S, v0, v1);
    store_tiles<T, ST>(k0 + K_S, k1 + K_S, S, s, v0, v1);
    store_one<T>(k0 + K_S + S, k1 + K_S + S, (T) 1, v0, v1);

    /* the normalised state and its gradient */
    T z[ST][4], n[ST][4], xh[ST][4];
    load_tiles2<P, ST>(z, save + sv0 + SP, save + sv1 + SP, S, v0, v1);
    load_tiles2<P, ST>(n, save + sv0 + 2 * SP, save + sv1 + 2 * SP, S, v0, v1);
    const T mu0 = v0 ? save[sv0 + 4 * SP + HP] : (T) 0, rstd0 = v0 ? save[sv0 + 4 * SP + HP + 1] : (T) 0;
    const T mu1 = v1 ? save[sv1 + 4 * SP + HP] : (T) 0, rstd1 = v1 ? save[sv1 + 4 * SP + HP + 1] : (T) 0;
    #pragma unroll
    for (int i = 0; i < ST; ++i) {
        int c = 8 * i + 2 * t;
        T h0 = ((T) 1 - z[i][0]) * n[i][0] + z[i][0] * s[i][0], h1 = ((T) 1 - z[i][1]) * n[i][1] + z[i][1] * s[i][1];
        T h2 = ((T) 1 - z[i][2]) * n[i][2] + z[i][2] * s[i][2], h3 = ((T) 1 - z[i][3]) * n[i][3] + z[i][3] * s[i][3];
        xh[i][0] = c < S ? (h0 - mu0) * rstd0 : (T) 0; xh[i][1] = c + 1 < S ? (h1 - mu0) * rstd0 : (T) 0;
        xh[i][2] = c < S ? (h2 - mu1) * rstd1 : (T) 0; xh[i][3] = c + 1 < S ? (h3 - mu1) * rstd1 : (T) 0;
    }
    {
        T y[ST][4];
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            int c = 8 * i + 2 * t;
            T g0 = c < S ? gamma[c] : (T) 0, g1 = c + 1 < S ? gamma[c + 1] : (T) 0;
            T e0 = c < S ? beta[c] : (T) 0, e1 = c + 1 < S ? beta[c + 1] : (T) 0;
            y[i][0] = xh[i][0] * g0 + e0; y[i][1] = xh[i][1] * g1 + e1;
            y[i][2] = xh[i][2] * g0 + e0; y[i][3] = xh[i][3] * g1 + e1;
        }
        store_tiles<T, ST>(k0 + K_Y, k1 + K_Y, S, y, v0, v1);
        store_one<T>(k0 + K_Y + S, k1 + K_Y + S, (T) 1, v0, v1);
    }
    T gy[ST][4];
    load_tiles<T, ST>(gy, g + base0 + O_SI, g + base1 + O_SI, S, v0, v1);
    {
        T gb[DT][4], more[ST][4];
        zero<T, DT>(gb);
        #pragma unroll 1
        for (int i = 0; i < LEGS; ++i) {
            const long dest = pinv[cell * CW + i * D];
            T leg[DT][4];
            load_tiles<T, DT>(leg, g + (long) r0 * TOTAL + dest, g + (long) r1 * TOTAL + dest, D, v0, v1);
            add_tiles<T, DT>(gb, leg);
        }
        store_tiles<T, DT>(k0 + K_B, k1 + K_B, D, gb, v0, v1);
        load_tiles<T, ST>(more, g + base0 + O_SO, g + base1 + O_SO, S, v0, v1);
        add_tiles<T, ST>(gy, more);
        R gba[DT][4];
        round_tiles<P, DT>(gba, gb);
        gemm<P, ST, DT>(gy, gba, weT, STR_ET, 0);
    }
    T gh[ST][4];
    {
        T gx[ST][4], gxx[ST][4];
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            int c = 8 * i + 2 * t;
            T g0 = c < S ? gamma[c] : (T) 0, g1 = c + 1 < S ? gamma[c + 1] : (T) 0;
            gx[i][0] = gy[i][0] * g0; gx[i][1] = gy[i][1] * g1; gx[i][2] = gy[i][2] * g0; gx[i][3] = gy[i][3] * g1;
            #pragma unroll
            for (int q = 0; q < 4; ++q) { gxx[i][q] = gy[i][q] * xh[i][q]; }
        }
        store_tiles<T, ST>(k0 + K_G, k1 + K_G, S, gxx, v0, v1);
        store_tiles<T, ST>(k0 + K_G + S, k1 + K_G + S, S, gy, v0, v1);
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) gxx[i][q] = gx[i][q] * xh[i][q];
        }
        T mgx0, mgx1, mgxx0, mgxx1;
        row_sums<T, ST>(mgx0, mgx1, gx, S);
        row_sums<T, ST>(mgxx0, mgxx1, gxx, S);
        mgx0 /= (T) S; mgx1 /= (T) S; mgxx0 /= (T) S; mgxx1 /= (T) S;
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            int c = 8 * i + 2 * t;
            gh[i][0] = c < S ? rstd0 * (gx[i][0] - mgx0 - xh[i][0] * mgxx0) : (T) 0;
            gh[i][1] = c + 1 < S ? rstd0 * (gx[i][1] - mgx0 - xh[i][1] * mgxx0) : (T) 0;
            gh[i][2] = c < S ? rstd1 * (gx[i][2] - mgx1 - xh[i][2] * mgxx1) : (T) 0;
            gh[i][3] = c + 1 < S ? rstd1 * (gx[i][3] - mgx1 - xh[i][3] * mgxx1) : (T) 0;
        }
    }

    /* the gates: g_s, g_az, g_an from z, n, s; g_hn, g_ar from r, hn */
    T gs[ST][4];
    R ga[4 * ST][4];
    {
        T gan[ST][4];
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                T ghq = gh[i][q], zq = z[i][q], nq = n[i][q];
                gs[i][q] = ghq * zq;
                gan[i][q] = ghq * ((T) 1 - zq) * ((T) 1 - nq * nq);
                gh[i][q] = ghq * (s[i][q] - nq) * zq * ((T) 1 - zq);
            }
        }
        store_tiles<T, ST>(k0 + S, k1 + S, S, gh, v0, v1);
        round_tiles<P, ST>(ga + ST, gh);
        T r[ST][4], hn[ST][4];
        load_tiles2<P, ST>(r, save + sv0, save + sv1, S, v0, v1);
        load_tiles2<P, ST>(hn, save + sv0 + 3 * SP, save + sv1 + 3 * SP, S, v0, v1);
        #pragma unroll
        for (int i = 0; i < ST; ++i) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                hn[i][q] = gan[i][q] * hn[i][q] * r[i][q] * ((T) 1 - r[i][q]);
                r[i][q] = gan[i][q] * r[i][q];
            }
        }
        store_tiles<T, ST>(k0, k1, S, hn, v0, v1);
        store_tiles<T, ST>(k0 + 2 * S, k1 + 2 * S, S, r, v0, v1);
        store_tiles<T, ST>(k0 + 3 * S, k1 + 3 * S, S, gan, v0, v1);
        round_tiles<P, ST>(ga, hn);
        round_tiles<P, ST>(ga + 2 * ST, r);
        round_tiles<P, ST>(ga + 3 * ST, gan);
    }
    gemm<P, ST, 3 * ST>(gs, ga, whhT, STR_HHT, 0);
    store_tiles<T, ST>(gin + base0 + O_SI, gin + base1 + O_SI, S, gs, v0, v1);
    zero<T, ST>(gs);
    store_tiles<T, ST>(gin + base0 + O_SO, gin + base1 + O_SO, S, gs, v0, v1);
    {
        T gp[HT][4];
        zero<T, HT>(gp);
        gemm<P, HT, 2 * ST>(gp, ga, wihT, STR_IHT, 0);
        gemm<P, HT, ST>(gp, ga + 3 * ST, wihT, STR_IHT, 2 * SP);
        store_tiles<T, HT>(pool + m0 * H, pool + m1 * H, H, gp, v0, v1);
    }
    {
        T gca[XT][4];
        zero<T, XT>(gca);
        gemm<P, XT, 2 * ST>(gca, ga, wihT + HP * STR_IHT, STR_IHT, 0);
        gemm<P, XT, ST>(gca, ga + 3 * ST, wihT + HP * STR_IHT, STR_IHT, 2 * SP);
        if (ECHO_C || ECHO_A) {
            T e[XT][4];
            load_given_tiles<T>(e, g + base0, g + base1, ECHO_C ? O_CI : O_AI, ECHO_A ? O_AI : O_CI, v0, v1);
            #pragma unroll
            for (int i = 0; i < XT; ++i) {
                int c = 8 * i + 2 * t;
                bool keep0 = c < C ? ECHO_C : ECHO_A, keep1 = c + 1 < C ? ECHO_C : ECHO_A;
                gca[i][0] += keep0 ? e[i][0] : (T) 0; gca[i][1] += keep1 ? e[i][1] : (T) 0;
                gca[i][2] += keep0 ? e[i][2] : (T) 0; gca[i][3] += keep1 ? e[i][3] : (T) 0;
            }
            load_given_tiles<T>(e, g + base0, g + base1, ECHO_C ? O_CO : O_AO, ECHO_A ? O_AO : O_CO, v0, v1);
            #pragma unroll
            for (int i = 0; i < XT; ++i) {
                int c = 8 * i + 2 * t;
                bool keep0 = c < C ? ECHO_C : ECHO_A, keep1 = c + 1 < C ? ECHO_C : ECHO_A;
                gca[i][0] += keep0 ? e[i][0] : (T) 0; gca[i][1] += keep1 ? e[i][1] : (T) 0;
                gca[i][2] += keep0 ? e[i][2] : (T) 0; gca[i][3] += keep1 ? e[i][3] : (T) 0;
            }
        }
        store_given_tiles<T>(gin + base0, gin + base1, O_CI, O_AI, gca, v0, v1);
        zero<T, XT>(gca);
        store_given_tiles<T>(gin + base0, gin + base1, O_CO, O_AO, gca, v0, v1);
    }
    }
}

constexpr int STR_W1T = STRIDE(HP), STR_W2T = STRIDE(HP);
constexpr int SCRATCH = 32 * 4;
constexpr int SM_CELL_BWD_ENCODE = HP * STR_W1 + (SP + DP) * STR_W1T + HP * STR_W2T + WARPS * HT * SCRATCH;
constexpr int K_L = LEGS * H, K_P = K_L + LEGS * D, K_H = K_P + H, KW_ENCODE = K_H + H + 1;

/*
cell_bwd_encode: the second half of the backward of cell_fwd, the
encoder, a warp per 16 rows of one cell on a persistent grid.  From
the pooled gradient of cell_bwd_gate and the incoming messages it
recomputes the first layer, writes the gradient of every leg to ``gin``
and adds the encoder's share to the state's, and writes per (row, cell)
the row of ``keep`` that fused._cell_grads reads: ``[gpre per leg | leg
per leg | gp | h1s, LEGS]`` where ``gp`` is the sum of the ``gpre``.
*/
template <int P> __global__ void __launch_bounds__(THREADS, P == TF32 ? 2 : 1) cell_bwd_encode(
        const typename Prec<P>::T* x, typename Prec<P>::T* gin, const typename Prec<P>::T* pool,
        typename Prec<P>::T* keep, int rows,
        const typename Prec<P>::T* w1, const typename Prec<P>::T* b1, const typename Prec<P>::T* w2) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::R R;
    extern __shared__ __align__(16) unsigned char smem[];
    R* w1s = (R*) smem;
    R* w1T = w1s + HP * STR_W1;
    R* w2T = w1T + (SP + DP) * STR_W1T;
    T* scratch = (T*) (w2T + HP * STR_W2T) + (threadIdx.x >> 5) * HT * SCRATCH;
    stage<P, HP, STR_W1, H, S, SP>(w1s, w1, S + D);
    stage<P, HP, STR_W1, H, D, DP>(w1s + SP, w1 + S, S + D);
    stage_t<P, SP, STR_W1T, S, H, HP>(w1T, w1, S + D);
    stage_t<P, DP, STR_W1T, D, H, HP>(w1T + SP * STR_W1T, w1 + S, S + D);
    stage_t<P, HP, STR_W2T, H, H, HP>(w2T, w2, H);
    __syncthreads();
    stage_bias<P, HP, STR_W1, H>(w1s + S, b1);
    __syncthreads();

    const int gl = lane_g(), items = ((rows + 15) / 16) * NC;
    #pragma unroll 1
    for (int item = first_item(); item < items; item += item_step()) {
    const int cell = item % NC;
    const int r0 = (item / NC) * 16 + gl, r1 = r0 + 8;
    const bool v0 = r0 < rows, v1 = r1 < rows;
    const long base0 = (long) r0 * TOTAL + cell * CW, base1 = (long) r1 * TOTAL + cell * CW;
    const long m0 = (long) r0 * NC + cell, m1 = (long) r1 * NC + cell;
    T* k0 = keep + m0 * KW_ENCODE;
    T* k1 = keep + m1 * KW_ENCODE;
    const T* p0 = x + base0;
    const T* p1 = x + base1;

    /* the gradient of the first layer's activations, parked in the warp's scratch */
    {
        T gh2[HT][4];
        load_tiles<T, HT>(gh2, pool + m0 * H, pool + m1 * H, H, v0, v1);
        #pragma unroll
        for (int i = 0; i < HT; ++i) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) gh2[i][q] /= (T) LEGS;
        }
        R gha[HT][4];
        round_tiles<P, HT>(gha, gh2);
        zero<T, HT>(gh2);
        gemm<P, HT, HT>(gh2, gha, w2T, STR_W2T, 0);
        stash<P, HT>(scratch, gh2);
    }
    R xa[ST + DT][4];
    load_a<P, ST>(xa, p0 + O_SI, p1 + O_SI, S, v0, v1);

    /* pass one: the summed activations */
    {
        T h1s[HT][4];
        zero<T, HT>(h1s);
        #pragma unroll 1
        for (int i = 0; i < LEGS; ++i) {
            T leg[DT][4];
            load_tiles<T, DT>(leg, p0 + i * D, p1 + i * D, D, v0, v1);
            store_tiles<T, DT>(k0 + K_L + i * D, k1 + K_L + i * D, D, leg, v0, v1);
            round_tiles<P, DT>(xa + ST, leg);
            set_one<P, DT>(xa + ST, D);
            T pre[HT][4];
            zero<T, HT>(pre);
            gemm<P, HT, ST + DT>(pre, xa, w1s, STR_W1, 0);
            #pragma unroll
            for (int j = 0; j < HT; ++j) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) h1s[j][q] += relu(pre[j][q]);
            }
        }
        store_tiles<T, HT>(k0 + K_H, k1 + K_H, H, h1s, v0, v1);
        store_one<T>(k0 + K_H + H, k1 + K_H + H, (T) LEGS, v0, v1);
    }

    /* pass two: the pre-activation gradients, the leg gradients and the state's */
    T gp[HT][4];
    zero<T, HT>(gp);
    #pragma unroll 1
    for (int i = 0; i < LEGS; ++i) {
        load_a<P, DT>(xa + ST, p0 + i * D, p1 + i * D, D, v0, v1);
        T pre[HT][4];
        zero<T, HT>(pre);
        gemm<P, HT, ST + DT>(pre, xa, w1s, STR_W1, 0);
        #pragma unroll
        for (int j = 0; j < HT; ++j) {
            T v[4];
            fetch<P>(scratch, j, v);
            #pragma unroll
            for (int q = 0; q < 4; ++q) pre[j][q] = pre[j][q] > (T) 0 ? v[q] : (T) 0;
        }
        store_tiles<T, HT>(k0 + i * H, k1 + i * H, H, pre, v0, v1);
        add_tiles<T, HT>(gp, pre);
        R gpa[HT][4];
        round_tiles<P, HT>(gpa, pre);
        T gl2[DT][4];
        zero<T, DT>(gl2);
        gemm<P, DT, HT>(gl2, gpa, w1T + SP * STR_W1T, STR_W1T, 0);
        store_tiles<T, DT>(gin + base0 + i * D, gin + base1 + i * D, D, gl2, v0, v1);
    }
    store_tiles<T, HT>(k0 + K_P, k1 + K_P, H, gp, v0, v1);
    {
        R gpa[HT][4];
        round_tiles<P, HT>(gpa, gp);
        T gs[ST][4];
        load_tiles<T, ST>(gs, gin + base0 + O_SI, gin + base1 + O_SI, S, v0, v1);
        gemm<P, ST, HT>(gs, gpa, w1T, STR_W1T, 0);
        store_tiles<T, ST>(gin + base0 + O_SI, gin + base1 + O_SI, S, gs, v0, v1);
    }
    }
}

constexpr int STR_R2T = STRIDE(DP), STR_R1T = STRIDE(UP), STR_PHIT = STRIDE(UP);
constexpr int SM_UNIT_BWD = UP * STR_PHI + UP * STR_R1 + UP * STR_R2T + (DP + UP) * STR_R1T + DP * STR_PHIT
    + WARPS * UT * SCRATCH;
constexpr int U_L = MEMBERS * HU, U_R = U_L + MEMBERS * (D + 1), U_H = U_R + MEMBERS * HU;
constexpr int U_O = U_H + MEMBERS * (HU + 1), U_P = U_O + MEMBERS * D, KW_UNIT = U_P + 2 * HU;

/*
unit_bwd: the backward of unit_fwd, a warp per 16 rows of one unit on a
persistent grid, recomputing the unit from its legs.  Writes the
gradient of every leg to ``gin`` and per (row, unit) the row of ``keep``
that fused_cuda._unit_grads reads: ``[gphi per member | leg, 1 per
member | g_rho per member | rho, 1 per member | g_out per member | pool
| g_sum]``.
*/
template <int P> __global__ void __launch_bounds__(THREADS, P == TF32 ? 2 : 1) unit_bwd(
        const typename Prec<P>::T* x, const typename Prec<P>::T* g, typename Prec<P>::T* gin,
        typename Prec<P>::T* keep, const long long* pinv, int rows,
        const typename Prec<P>::T* wphi, const typename Prec<P>::T* bphi,
        const typename Prec<P>::T* wr1, const typename Prec<P>::T* br1, const typename Prec<P>::T* wr2) {
    typedef typename Prec<P>::T T; typedef typename Prec<P>::R R;
    extern __shared__ __align__(16) unsigned char smem[];
    R* wphis = (R*) smem;
    R* wr1s = wphis + UP * STR_PHI;
    R* wr2T = wr1s + UP * STR_R1;
    R* wr1T = wr2T + UP * STR_R2T;
    R* wphiT = wr1T + (DP + UP) * STR_R1T;
    T* scratch = (T*) (wphiT + DP * STR_PHIT) + (threadIdx.x >> 5) * UT * SCRATCH;
    stage<P, UP, STR_PHI, HU, D, DP>(wphis, wphi, D);
    stage<P, UP, STR_R1, HU, D, DP>(wr1s, wr1, D + HU);
    stage<P, UP, STR_R1, HU, HU, UP>(wr1s + DP, wr1 + D, D + HU);
    stage_t<P, UP, STR_R2T, HU, D, DP>(wr2T, wr2, HU);
    stage_t<P, DP, STR_R1T, D, HU, UP>(wr1T, wr1, D + HU);
    stage_t<P, UP, STR_R1T, HU, HU, UP>(wr1T + DP * STR_R1T, wr1 + D, D + HU);
    stage_t<P, DP, STR_PHIT, D, HU, UP>(wphiT, wphi, D);
    __syncthreads();
    stage_bias<P, UP, STR_PHI, HU>(wphis + D, bphi);
    stage_bias<P, UP, STR_R1, HU>(wr1s + D, br1);
    __syncthreads();

    const int gl = lane_g(), items = ((rows + 15) / 16) * NU;
    #pragma unroll 1
    for (int item = first_item(); item < items; item += item_step()) {
    const int unit = item % NU;
    const int r0 = (item / NU) * 16 + gl, r1 = r0 + 8;
    const bool v0 = r0 < rows, v1 = r1 < rows;
    const long base0 = (long) r0 * TOTAL + UOFF + unit * UW, base1 = (long) r1 * TOTAL + UOFF + unit * UW;
    const long m0 = (long) r0 * NU + unit, m1 = (long) r1 * NU + unit;
    T* k0 = keep + m0 * KW_UNIT;
    T* k1 = keep + m1 * KW_UNIT;
    const T* p0 = x + base0;
    const T* p1 = x + base1;

    /* the pooled embedding, and the pooled half of rho's first layer parked in scratch */
    {
        T pool[UT][4];
        zero<T, UT>(pool);
        #pragma unroll 1
        for (int i = 0; i < MEMBERS; ++i) {
            T leg[DT][4];
            load_tiles<T, DT>(leg, p0 + i * D, p1 + i * D, D, v0, v1);
            store_tiles<T, DT>(k0 + U_L + i * (D + 1), k1 + U_L + i * (D + 1), D, leg, v0, v1);
            store_one<T>(k0 + U_L + i * (D + 1) + D, k1 + U_L + i * (D + 1) + D, (T) 1, v0, v1);
            R la[DT][4];
            round_tiles<P, DT>(la, leg);
            set_one<P, DT>(la, D);
            T h[UT][4];
            zero<T, UT>(h);
            gemm<P, UT, DT>(h, la, wphis, STR_PHI, 0);
            #pragma unroll
            for (int j = 0; j < UT; ++j) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) pool[j][q] += relu(h[j][q]);
            }
        }
        store_tiles<T, UT>(k0 + U_P, k1 + U_P, HU, pool, v0, v1);
        R pa[UT][4];
        round_tiles<P, UT>(pa, pool);
        zero<T, UT>(pool);
        gemm<P, UT, UT>(pool, pa, wr1s, STR_R1, DP);
        stash<P, UT>(scratch, pool);
    }

    /* rho's gradients per member, and the gradient of the pooled embedding */
    T gsum[UT][4];
    zero<T, UT>(gsum);
    #pragma unroll 1
    for (int i = 0; i < MEMBERS; ++i) {
        R la[DT][4];
        load_a<P, DT>(la, p0 + i * D, p1 + i * D, D, v0, v1);
        const long dest = pinv[UOFF + unit * UW + i * D];
        T gr[UT][4];
        {
            T go[DT][4];
            load_tiles<T, DT>(go, g + (long) r0 * TOTAL + dest, g + (long) r1 * TOTAL + dest, D, v0, v1);
            store_tiles<T, DT>(k0 + U_O + i * D, k1 + U_O + i * D, D, go, v0, v1);
            R goa[DT][4];
            round_tiles<P, DT>(goa, go);
            zero<T, UT>(gr);
            gemm<P, UT, DT>(gr, goa, wr2T, STR_R2T, 0);
        }
        #pragma unroll
        for (int j = 0; j < UT; ++j) {
            T p[4];
            fetch<P>(scratch, j, p);
            #pragma unroll
            for (int q = 0; q < DT; ++q) {
                R b[2];
                frag_b<P>(b, wr1s, STR_R1, 8 * j, 8 * q);
                mma<P>(p, la[q], b);
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) { gr[j][q] = p[q] > (T) 0 ? gr[j][q] : (T) 0; p[q] = relu(p[q]); }
            store_tile<T>(k0 + U_H + i * (HU + 1), k1 + U_H + i * (HU + 1), 8 * j, HU, p, v0, v1);
        }
        store_one<T>(k0 + U_H + i * (HU + 1) + HU, k1 + U_H + i * (HU + 1) + HU, (T) 1, v0, v1);
        store_tiles<T, UT>(k0 + U_R + i * HU, k1 + U_R + i * HU, HU, gr, v0, v1);
        add_tiles<T, UT>(gsum, gr);
        R gra[UT][4];
        round_tiles<P, UT>(gra, gr);
        T gl2[DT][4];
        zero<T, DT>(gl2);
        gemm<P, DT, UT>(gl2, gra, wr1T, STR_R1T, 0);
        store_tiles<T, DT>(gin + base0 + i * D, gin + base1 + i * D, D, gl2, v0, v1);
    }
    store_tiles<T, UT>(k0 + U_P + HU, k1 + U_P + HU, HU, gsum, v0, v1);
    T gpool[UT][4];
    {
        R gsa[UT][4];
        round_tiles<P, UT>(gsa, gsum);
        zero<T, UT>(gpool);
        gemm<P, UT, UT>(gpool, gsa, wr1T + DP * STR_R1T, STR_R1T, 0);
    }

    /* phi's gradients per member */
    #pragma unroll 1
    for (int i = 0; i < MEMBERS; ++i) {
        R la[DT][4];
        load_a<P, DT>(la, p0 + i * D, p1 + i * D, D, v0, v1);
        T gphi[UT][4];
        #pragma unroll
        for (int j = 0; j < UT; ++j) {
            T p[4] = {(T) 0, (T) 0, (T) 0, (T) 0};
            #pragma unroll
            for (int q = 0; q < DT; ++q) {
                R b[2];
                frag_b<P>(b, wphis, STR_PHI, 8 * j, 8 * q);
                mma<P>(p, la[q], b);
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) gphi[j][q] = p[q] > (T) 0 ? gpool[j][q] : (T) 0;
        }
        store_tiles<T, UT>(k0 + i * HU, k1 + i * HU, HU, gphi, v0, v1);
        R gpa[UT][4];
        round_tiles<P, UT>(gpa, gphi);
        T gl2[DT][4];
        load_tiles<T, DT>(gl2, gin + base0 + i * D, gin + base1 + i * D, D, v0, v1);
        gemm<P, DT, UT>(gl2, gpa, wphiT, STR_PHIT, 0);
        store_tiles<T, DT>(gin + base0 + i * D, gin + base1 + i * D, D, gl2, v0, v1);
    }
    }
}
