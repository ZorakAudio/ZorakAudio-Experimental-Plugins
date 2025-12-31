import("stdfaust.lib");

declare name "ModTilt (SAFE) - envelope tilt shaper";
declare version "1.1";

// ---------------- UI ----------------
// ---------------- UI (ordered) ----------------
tilt_db = vgroup("Control",
  hslider("1. Tilt (dB)", 0.0, -6.0, 3.0, 0.1)
);

pivotHz = vgroup("Control",
  hslider("2. Pivot (Hz)", 3.0, 2.0, 5.0, 0.01)
);

mix = vgroup("Mix",
  hslider("1. Mix", 1.0, 0.0, 1.0, 0.001)
);

// ---------------- helpers ----------------
max2(x,y) = x,y : max;

// a_from_hz(hz) = 1 - exp(-2*pi*max(hz,0.001)/SR)
a_from_hz(hz) =
  (-2.0*ma.PI*max2(hz,0.001)/ma.SR) : exp : (1.0 - _);

// one-pole smoother: y = a*x + (1-a)*y(z^-1)
onepole(a) = *(a) : (+ ~ *(1.0 - a));

clamp(x, lo, hi) = min(max(x, lo), hi);

// ---------------- locked SAFE internals ----------------
a_env   = a_from_hz(25.0);     // env detector Hz
a_base  = a_from_hz(1.0);      // baseline Hz
a_piv   = a_from_hz(pivotHz);  // pivot LPF Hz

a_ratio = 0.05;                // ratio smoothing (empirical, stable)
a_trim  = a_from_hz(0.2);      // auto-trim speed

depth   = 0.75;

g_hi = ba.db2linear( tilt_db * 0.5 );
g_lo = ba.db2linear(-tilt_db * 0.5 );

// ---------------- core ----------------
process = stereo;

stereo(xL, xR) = (outL, outR)
with {
  // linked stereo detect
  x  = 0.5*(xL + xR);

  // envelope RMS-ish
  x2   = x*x;
  env2 = x2 : onepole(a_env);
  env  = sqrt(max(env2, 0.0));

  // baseline
  base = env : onepole(a_base);

  // modulation
  m = env - base;

  // split modulation around pivot
  lp_piv = m : onepole(a_piv);
  m_lo   = lp_piv;
  m_hi   = m - lp_piv;

  // tilt + depth blend
  m2_tilt = (m_lo * g_lo) + (m_hi * g_hi);
  m2      = m*(1.0 - depth) + m2_tilt*depth;

  // recombine
  env_t = base + m2;

  // NaN-proof ratio + relative floor
  env_tp = max(env_t, 0.05*env);
  eps    = 1e-9;
  r0     = (env_tp + eps) / (env + eps);

  // hard clamp (safe range)
  r0c = clamp(r0, 0.67, 1.5);

  // ratio smoothing with init=1
  r_s = 1.0 + ((r0c - 1.0) : onepole(a_ratio));
  g   = r_s;

  // auto-trim
  rdb      = 20.0 * log10(max(r_s, 1e-12));
  mean_rdb = rdb : onepole(a_trim);
  trim     = ba.db2linear(-mean_rdb);

  // apply to audio
  yL = xL * g;
  yR = xR * g;

  outL = (xL*(1.0 - mix) + yL*mix) * trim;
  outR = (xR*(1.0 - mix) + yR*mix) * trim;
};
