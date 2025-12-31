declare filename "untitled.dsp";
declare name "untitled";
declare name "Vocal Air Restore (LoG HF Expander)";
declare description "Stereo 'air' restoration: HF carrier expansion + band-limited noise, driven by a level-invariant LoG-ish curvature detector.";
declare author "Converted from JSFX -> Faust (with perceptual/robustness tweaks)";
declare license "MIT";

import("stdfaust.lib");

// ---------- helpers ----------
eps = 1e-12;

// Keep any fixed frequency safely below Nyquist for low sample rates
safeFc(fc) = min(fc, 0.45*ma.SR);

// JSFX-style RBJ biquad coefficient builder + direct-form-II transposed section (tf22t)
rbjHP(fc,Q) = fi.tf22t(b0,b1,b2,a1,a2) with {
  f  = safeFc(fc);
  q  = max(0.001, Q);
  w0 = 2.0*ma.PI*f/ma.SR;
  cw = cos(w0);
  sw = sin(w0);
  alpha = sw/(2.0*q);

  bb0 = (1.0+cw)/2.0;
  bb1 = -(1.0+cw);
  bb2 = (1.0+cw)/2.0;

  aa0 = 1.0+alpha;
  aa1 = -2.0*cw;
  aa2 = 1.0-alpha;

  b0 = bb0/aa0; b1 = bb1/aa0; b2 = bb2/aa0;
  a1 = aa1/aa0; a2 = aa2/aa0;
};

rbjLP(fc,Q) = fi.tf22t(b0,b1,b2,a1,a2) with {
  f  = safeFc(fc);
  q  = max(0.001, Q);
  w0 = 2.0*ma.PI*f/ma.SR;
  cw = cos(w0);
  sw = sin(w0);
  alpha = sw/(2.0*q);

  bb0 = (1.0-cw)/2.0;
  bb1 = 1.0-cw;
  bb2 = (1.0-cw)/2.0;

  aa0 = 1.0+alpha;
  aa1 = -2.0*cw;
  aa2 = 1.0-alpha;

  b0 = bb0/aa0; b1 = bb1/aa0; b2 = bb2/aa0;
  a1 = aa1/aa0; a2 = aa2/aa0;
};

// JSFX used a "BPF (constant skirt)" variant using b0=sw/2 (NOT alpha). Keep that behavior.
rbjBP_skirt(fc,Q) = fi.tf22t(b0,b1,b2,a1,a2) with {
  f  = safeFc(fc);
  q  = max(0.001, Q);
  w0 = 2.0*ma.PI*f/ma.SR;
  cw = cos(w0);
  sw = sin(w0);
  alpha = sw/(2.0*q);

  bb0 = sw/2.0;
  bb1 = 0.0;
  bb2 = -sw/2.0;

  aa0 = 1.0+alpha;
  aa1 = -2.0*cw;
  aa2 = 1.0-alpha;

  b0 = bb0/aa0; b1 = bb1/aa0; b2 = bb2/aa0;
  a1 = aa1/aa0; a2 = aa2/aa0;
};

// One-pole lowpass parameterized by exponential coefficient 'a' (JSFX form):
// y = x + (y_prev - x)*a  == (1-a)*x + a*y_prev
onePoleExp(a) = *(1.0-a) : (+ ~ *(a));

pow1(x,p) = pow(max(eps,x), p);

// ---------- UI ----------
amount = hslider("Air Amount [%]", 35, 0, 100, 1)/100.0 : si.smoo;
sens   = hslider("Sensitivity [%]", 50, 0, 100, 1)/100.0 : si.smoo;
trimDB = 0;

outGain = ba.db2linear(trimDB);

// Hard safety caps (match JSFX intent)
maxExp_dB  = 5.0 * amount;                 // 0..+5 dB
maxExp_lin = ba.db2linear(maxExp_dB);
airMix     = 0.25 * amount;                // 0..0.25 effective
airBase    = ba.db2linear(-34.0);           // conservative base level

// Sensitivity maps to a normalized-curvature threshold (dimensionless)
thrN = 0.18 - 0.13*sens;                    // sens 0 -> 0.18, sens 1 -> 0.05

// ---------- fixed DSP constants ----------
det_fc = 9500.0; det_Q = 1.0;
detSmooth_fc = 8500.0;
detSmooth_a  = exp(-2.0*ma.PI*safeFc(detSmooth_fc)/ma.SR);

hf_fc  = 11500.0; hf_Q  = 0.707;
air_fc = 16000.0; air_Q = 1.2;

// Envelope AR for normalized curvature
atk_s = 0.0025;
rel_s = 0.080;

// ---------- main ----------
process(inL, inR) = (outL, outR) with {

  // --- detector BPF ---
  detL = inL : rbjBP_skirt(det_fc, det_Q);
  detR = inR : rbjBP_skirt(det_fc, det_Q);

  // --- 2-stage smoothing (for LoG-ish curvature) ---
  sm2L = detL : onePoleExp(detSmooth_a) : onePoleExp(detSmooth_a);
  sm2R = detR : onePoleExp(detSmooth_a) : onePoleExp(detSmooth_a);

  // --- LoG-ish Laplacian (2nd difference) ---
  s0L = sm2L;
  s1L = sm2L';
  s2L = sm2L'';
  lapL = s0L - 2.0*s1L + s2L;

  s0R = sm2R;
  s1R = sm2R';
  s2R = sm2R'';
  lapR = s0R - 2.0*s1R + s2R;

  denomL = abs(s0L) + 2.0*abs(s1L) + abs(s2L) + eps;
  denomR = abs(s0R) + 2.0*abs(s1R) + abs(s2R) + eps;

  curvN_L = abs(lapL)/denomL;
  curvN_R = abs(lapR)/denomR;
  curvN   = 0.5*(curvN_L + curvN_R);

  // AR-smoothed normalized curvature
  env = curvN : si.onePoleSwitching(atk_s, rel_s);

  // Soft trigger mapping (same shape as JSFX)
  u = max(0.0, env/thrN - 1.0);
  t = u/(1.0+u);
  t2 = pow1(t, 1.8);           // late ramp -> less fizz

  // Bounded HF expansion gain
  g = 1.0 + t*(maxExp_lin - 1.0);

  // --- real HF carrier: 4th-order HPF (2 cascaded biquads) ---
  hfL = inL : rbjHP(hf_fc, hf_Q) : rbjHP(hf_fc, hf_Q);
  hfR = inR : rbjHP(hf_fc, hf_Q) : rbjHP(hf_fc, hf_Q);

  hfAddL = hfL * (g - 1.0);
  hfAddR = hfR * (g - 1.0);

  // --- synthetic air halo ---
  nL = no.noise;
  nR = no.noise;

  airL = nL : rbjBP_skirt(air_fc, air_Q);
  airR = nR : rbjBP_skirt(air_fc, air_Q);

  airGain = (t2 * airBase) * airMix;
  airAddL = airL * airGain;
  airAddR = airR * airGain;

  outL = (inL + hfAddL + airAddL) * outGain;
  outR = (inR + hfAddR + airAddR) * outGain;
};
