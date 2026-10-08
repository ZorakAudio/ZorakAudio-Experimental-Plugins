// Shared production FFT/convolution/memory implementations. Include after generated state and WDL fft.h.
#pragma once
namespace
{
#if ! defined (ZA_JSFX_FFT_LEGACY_IN_ORDER)
 #define ZA_JSFX_FFT_LEGACY_IN_ORDER 0
#endif

static_assert (sizeof (WDL_FFT_COMPLEX) == sizeof (double) * 2,
               "WDL_FFT_COMPLEX must map to interleaved doubles");
static_assert (std::is_trivially_copyable_v<WDL_FFT_COMPLEX>,
               "WDL_FFT_COMPLEX must be trivially copyable");

static inline bool isPowerOfTwo (int64_t n) noexcept
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

static constexpr int64_t kJsfxFftMinSize = 16;
static constexpr int64_t kJsfxFftMaxSize = 32768;
static constexpr int64_t kJsfxFftPageDoubles = 65536;

struct JsfxFftScratchBuffer
{
    WDL_FFT_COMPLEX* data = nullptr;
    int capacity = 0;

    ~JsfxFftScratchBuffer()
    {
        std::free (data);
    }

    WDL_FFT_COMPLEX* ensure (int needed) noexcept
    {
        if (needed <= capacity)
            return data;

        void* p = std::realloc (data, (size_t) needed * sizeof (WDL_FFT_COMPLEX));
        if (p == nullptr)
            return nullptr;

        data = static_cast<WDL_FFT_COMPLEX*> (p);
        capacity = needed;
        return data;
    }
};

static thread_local JsfxFftScratchBuffer gJsfxFftScratch;

static inline int64_t jsfxRoundToIndex (double v) noexcept
{
    return (int64_t) (v + (v >= 0.0 ? 1.0e-5 : -1.0e-5));
}

static inline bool isSupportedJsfxFftSize (int64_t n) noexcept
{
    return n >= kJsfxFftMinSize && n <= kJsfxFftMaxSize && isPowerOfTwo (n);
}

static inline bool staysWithinJsfxPage (int64_t base, int64_t spanDoubles) noexcept
{
    if (base < 0 || spanDoubles <= 0)
        return false;

    const int64_t lastIndex = base + spanDoubles - 1;
    if (lastIndex < base)
        return false;

    return (base / kJsfxFftPageDoubles) == (lastIndex / kJsfxFftPageDoubles);
}

static inline bool staysWithinJsfxFftPage (int64_t base, int64_t complexCount) noexcept
{
    if (complexCount <= 0 || complexCount > (kJsfxFftPageDoubles / 2))
        return false;

    return staysWithinJsfxPage (base, 2 * complexCount);
}

static inline void ensureWdlFftInit() noexcept
{
    static std::once_flag once;
    std::call_once (once, [] { WDL_fft_init(); });
}

static bool prepareJsfxComplexBufferRegion (DSPJSFX_State* st, double baseD, double complexCountD,
                                            int64_t& base, int& complexCount) noexcept
{
    if (st == nullptr)
        return false;

    const int64_t count = jsfxRoundToIndex (complexCountD);
    int64_t start = jsfxRoundToIndex (baseD);
    if (start < 0)
        start = 0;

    if (count <= 0 || count > (kJsfxFftPageDoubles / 2))
        return false;

    if (! staysWithinJsfxFftPage (start, count))
        return false;

    const int64_t needed = start + (2 * count);
    jsfx_ensure_mem (st, needed);
    if (st->mem == nullptr || needed > st->memN)
        return false;

    base = start;
    complexCount = (int) count;
    return true;
}

static bool prepareJsfxFftRegion (DSPJSFX_State* st, double baseD, double sizeD, int64_t& base, int& N) noexcept
{
    const int64_t size = jsfxRoundToIndex (sizeD);
    if (! isSupportedJsfxFftSize (size))
        return false;

    if (! prepareJsfxComplexBufferRegion (st, baseD, (double) size, base, N))
        return false;

    ensureWdlFftInit();
    return true;
}

static bool prepareJsfxRealFftRegion (DSPJSFX_State* st, double baseD, double sizeD, int64_t& base, int& N) noexcept
{
    if (st == nullptr)
        return false;

    const int64_t size = jsfxRoundToIndex (sizeD);
    int64_t start = jsfxRoundToIndex (baseD);
    if (start < 0)
        start = 0;

    if (! isSupportedJsfxFftSize (size))
        return false;

    if (! staysWithinJsfxPage (start, size))
        return false;

    const int64_t needed = start + size;
    jsfx_ensure_mem (st, needed);
    if (st->mem == nullptr || needed > st->memN)
        return false;

    ensureWdlFftInit();

    base = start;
    N = (int) size;
    return true;
}

// FFT arithmetic operates on private storage in shared-state mode. Gather and
// commit are individual cell accesses, never an atomic transaction or UI frame.
struct JsfxFftAccess {
    DSPJSFX_Cell* shared;
    double* data;
    int count;
    JsfxFftAccess(DSPJSFX_Cell* cells, int n, int bank = 0) : shared(cells), count(n) {
#if DSPJSFX_NATIVE_GFX_LEGACY
        static thread_local std::array<std::array<double, 65536>, 2> scratch;
        data = scratch[(size_t) bank].data();
        jsfxCopyCells(data, cells, (size_t)n);
#else
        juce::ignoreUnused(bank);
        data = cells;
#endif
    }
    void commit() noexcept {
#if DSPJSFX_NATIVE_GFX_LEGACY
        jsfxWriteCells(shared, data, (size_t)count);
#endif
    }
};

static inline WDL_FFT_COMPLEX* asWdlComplex (double* interleaved) noexcept
{
    return reinterpret_cast<WDL_FFT_COMPLEX*> (interleaved);
}

static bool permuteWdlToNaturalInPlace (double* interleaved, int N) noexcept
{
    WDL_FFT_COMPLEX* scratch = gJsfxFftScratch.ensure (N);
    if (scratch == nullptr)
        return false;

    auto* buf = asWdlComplex (interleaved);
    const int* perm = WDL_fft_permute_tab (N);
    if (perm == nullptr)
        return false;

    for (int i = 0; i < N; ++i)
        scratch[(size_t) i] = buf[perm[i]];

    std::memcpy (buf, scratch, (size_t) N * sizeof (WDL_FFT_COMPLEX));
    return true;
}

static bool permuteNaturalToWdlInPlace (double* interleaved, int N) noexcept
{
    WDL_FFT_COMPLEX* scratch = gJsfxFftScratch.ensure (N);
    if (scratch == nullptr)
        return false;

    auto* buf = asWdlComplex (interleaved);
    const int* perm = WDL_fft_permute_tab (N);
    if (perm == nullptr)
        return false;

    for (int i = 0; i < N; ++i)
        scratch[(size_t) perm[i]] = buf[i];

    std::memcpy (buf, scratch, (size_t) N * sizeof (WDL_FFT_COMPLEX));
    return true;
}
} // namespace

extern "C" double jsfx_fft (DSPJSFX_State* st, double baseD, double sizeD)
{
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    if (gJsfxFftScratch.ensure (N) == nullptr)
        return 0.0;
#endif

    JsfxFftAccess access (st->mem + base, 2 * N);
    WDL_fft (asWdlComplex (access.data), N, 0);

#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    (void) permuteWdlToNaturalInPlace (access.data, N);
#endif

    access.commit();
    return 0.0;
}

extern "C" double jsfx_ifft (DSPJSFX_State* st, double baseD, double sizeD)
{
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

    JsfxFftAccess access (st->mem + base, 2 * N);
#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    if (! permuteNaturalToWdlInPlace (access.data, N))
        return 0.0;
#endif

    WDL_fft (asWdlComplex (access.data), N, 1);
    access.commit();
    return 0.0;
}

extern "C" double jsfx_fft_real (DSPJSFX_State* st, double baseD, double sizeD)
{
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxRealFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    if (gJsfxFftScratch.ensure (N / 2) == nullptr)
        return 0.0;
#endif

    JsfxFftAccess access (st->mem + base, N);
    WDL_real_fft (access.data, N, 0);

#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    (void) permuteWdlToNaturalInPlace (access.data, N / 2);
#endif

    access.commit();
    return 0.0;
}

extern "C" double jsfx_ifft_real (DSPJSFX_State* st, double baseD, double sizeD)
{
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxRealFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

    JsfxFftAccess access (st->mem + base, N);
#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    if (! permuteNaturalToWdlInPlace (access.data, N / 2))
        return 0.0;
#endif

    WDL_real_fft (access.data, N, 1);
    access.commit();
    return 0.0;
}

extern "C" double jsfx_convolve_c (DSPJSFX_State* st, double destD, double srcD, double sizeD)
{
    int64_t destBase = 0;
    int64_t srcBase = 0;
    int N = 0;
    int srcN = 0;
    if (! prepareJsfxComplexBufferRegion (st, destD, sizeD, destBase, N))
        return 0.0;
    if (! prepareJsfxComplexBufferRegion (st, srcD, sizeD, srcBase, srcN))
        return 0.0;
    if (srcN != N)
        return 0.0;

    JsfxFftAccess destAccess (st->mem + destBase, 2 * N);
    JsfxFftAccess srcAccess (st->mem + srcBase, 2 * N, 1);
    auto* dest = asWdlComplex (destAccess.data);
    auto* src = asWdlComplex (srcAccess.data);

    const int64_t destEnd = destBase + (2 * (int64_t) N);
    const int64_t srcEnd = srcBase + (2 * (int64_t) srcN);
    const bool overlaps = (destBase < srcEnd) && (srcBase < destEnd);

    if (overlaps && destBase != srcBase)
    {
        WDL_FFT_COMPLEX* scratch = gJsfxFftScratch.ensure (N);
        if (scratch == nullptr)
            return 0.0;

        std::memcpy (scratch, src, (size_t) N * sizeof (WDL_FFT_COMPLEX));
        src = scratch;
    }

    for (int i = 0; i < N; ++i)
    {
        const auto ar = dest[(size_t) i].re;
        const auto ai = dest[(size_t) i].im;
        const auto br = src[(size_t) i].re;
        const auto bi = src[(size_t) i].im;

        dest[(size_t) i].re = ar * br - ai * bi;
        dest[(size_t) i].im = ar * bi + ai * br;
    }

    destAccess.commit();
    return 0.0;
}

extern "C" double jsfx_memcpy (DSPJSFX_State* st, double destD, double srcD, double lengthD)
{
    if (st == nullptr)
        return 0.0;

    int64_t dest = jsfxRoundToIndex (destD);
    int64_t src = jsfxRoundToIndex (srcD);
    int64_t length = jsfxRoundToIndex (lengthD);

    if (dest < 0)
        dest = 0;
    if (src < 0)
        src = 0;
    if (length <= 0)
        return 0.0;

    const int64_t destEnd = dest + length;
    const int64_t srcEnd = src + length;
    if (destEnd < dest || srcEnd < src)
        return 0.0;

    const int64_t needed = std::max (destEnd, srcEnd);
    jsfx_ensure_mem (st, needed);
    if (st->mem == nullptr || needed > st->memN)
        return 0.0;

#if DSPJSFX_NATIVE_GFX_LEGACY
    if (dest > src && dest < srcEnd)
        for (int64_t i = length; i-- > 0;) st->mem[dest + i] = st->mem[src + i];
    else
        for (int64_t i = 0; i < length; ++i) st->mem[dest + i] = st->mem[src + i];
#else
    std::memmove (st->mem + dest, st->mem + src, (size_t) length * sizeof (double));
#endif
    return 0.0;
}



extern "C" double jsfx_fft_permute (DSPJSFX_State* st, double baseD, double sizeD)
{
#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    juce::ignoreUnused (st, baseD, sizeD);
    return 0.0;
#else
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

    JsfxFftAccess access (st->mem + base, 2 * N);
    (void) permuteWdlToNaturalInPlace (access.data, N);
    access.commit();
    return 0.0;
#endif
}

extern "C" double jsfx_fft_ipermute (DSPJSFX_State* st, double baseD, double sizeD)
{
#if ZA_JSFX_FFT_LEGACY_IN_ORDER
    juce::ignoreUnused (st, baseD, sizeD);
    return 0.0;
#else
    int64_t base = 0;
    int N = 0;
    if (! prepareJsfxFftRegion (st, baseD, sizeD, base, N))
        return 0.0;

    JsfxFftAccess access (st->mem + base, 2 * N);
    (void) permuteNaturalToWdlInPlace (access.data, N);
    access.commit();
    return 0.0;
#endif
}




