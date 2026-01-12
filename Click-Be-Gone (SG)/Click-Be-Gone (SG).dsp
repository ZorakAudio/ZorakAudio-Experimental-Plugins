import("stdfaust.lib");

declare name "Click-Be-Gone (SG)";
declare version "1.2";
declare description "Remove HF needle-clicks + small splats from wet granular without dulling texture. Modes are behavior presets (window ladder + gating + hold + max replace).";
declare author "Zorak Audio";
declare license "MIT License";
declare latency_sec "0";


amount = hslider("Amount [%]", 50, 0, 100, 1) / 100;
sensitivity = hslider("Sensitivity [%]", 50, 0, 100, 1) / 100;
hpf_hz = hslider("HPF [Hz]", 1500, 300, 6000, 10);
mode    = nentry("Mode[style:menu{'Fast':0;'Medium':1;'Slow':2}]", 1, 0, 2, 1);
monitor = nentry("Monitor[style:menu{'Output':0;'Delta':1}]",      0, 0, 1, 1);


eps = 1e-12;

ratio_thr0 = 6.0 - 4.0 * sensitivity;     // 6..2
err_thr0   = 0.25 - 0.17 * sensitivity;   // 0.25..0.08

ratio_mul = ba.selectn(3, mode, 1.12, 1.00, 0.92);
err_mul   = ba.selectn(3, mode, 1.18, 1.00, 0.90);
mix_mul   = ba.selectn(3, mode, 0.85, 1.00, 1.08);
hold_mul  = ba.selectn(3, mode, 0.75, 1.00, 1.35);

env_rel_ms0 = 30.0 - 20.0 * sensitivity;     // 30..10
base_ms0    = 300.0 - 180.0 * sensitivity;   // 300..120

base_mul = ba.selectn(3, mode, 0.85, 1.00, 1.10);
env_mul  = ba.selectn(3, mode, 0.85, 1.00, 1.10);
env_rel_ms = env_rel_ms0 * env_mul;
base_ms    = base_ms0    * base_mul;

ratio_thr = ratio_thr0 * ratio_mul;
err_thr   = err_thr0 * err_mul;

mix_max0 = 0.60 + 0.32 * amount; // 0.60..0.92
mix_max  = min(mix_max0 * mix_mul, 0.96);

holdN_base = 8 + (amount * 32);
holdN = max(holdN_base * hold_mul, 4);

env_rel = exp(-1000 / (ma.SR * env_rel_ms));
base_a  = 1 - exp(-1000 / (ma.SR * base_ms));

a = exp(-2 * ma.PI * hpf_hz / ma.SR);

clamp(x, a, b) = min(max(x, a), b);

sg11_pred(sig) = (-36 * (sig@(20)) + 9 * (sig@(19)) + 44 * (sig@(18)) + 69 * (sig@(17)) + 84 * (sig@(16)) + 89 * (sig@(15)) + 84 * (sig@(14)) + 69 * (sig@(13)) + 44 * (sig@(12)) + 9 * (sig@(11)) - 36 * (sig@(10))) / 429;

sg15_pred(sig) = (-78 * (sig@(22)) - 13 * (sig@(21)) + 42 * (sig@(20)) + 87 * (sig@(19)) + 122 * (sig@(18)) + 147 * (sig@(17)) + 162 * (sig@(16)) + 167 * (sig@(15)) + 162 * (sig@(14)) + 147 * (sig@(13)) + 122 * (sig@(12)) + 87 * (sig@(11)) + 42 * (sig@(10)) - 13 * (sig@(9)) - 78 * (sig@(8))) / 1105;

sg21_pred(sig) = (-171 * (sig@(25)) - 76 * (sig@(24)) + 9 * (sig@(23)) + 84 * (sig@(22)) + 149 * (sig@(21)) + 204 * (sig@(20)) + 249 * (sig@(19)) + 284 * (sig@(18)) + 309 * (sig@(17)) + 324 * (sig@(16)) + 329 * (sig@(15)) + 324 * (sig@(14)) + 309 * (sig@(13)) + 284 * (sig@(12)) + 249 * (sig@(11)) + 204 * (sig@(10)) + 149 * (sig@(9)) + 84 * (sig@(8)) + 9 * (sig@(7)) - 76 * (sig@(6)) - 171 * (sig@(5))) / 3059;

sg31_pred(sig) = (-406 * (sig@(30)) - 261 * (sig@(29)) - 126 * (sig@(28)) - 1 * (sig@(27)) + 114 * (sig@(26)) + 219 * (sig@(25)) + 314 * (sig@(24)) + 399 * (sig@(23)) + 474 * (sig@(22)) + 539 * (sig@(21)) + 594 * (sig@(20)) + 639 * (sig@(19)) + 674 * (sig@(18)) + 699 * (sig@(17)) + 714 * (sig@(16)) + 719 * (sig@(15)) + 714 * (sig@(14)) + 699 * (sig@(13)) + 674 * (sig@(12)) + 639 * (sig@(11)) + 594 * (sig@(10)) + 539 * (sig@(9)) + 474 * (sig@(8)) + 399 * (sig@(7)) + 314 * (sig@(6)) + 219 * (sig@(5)) + 114 * (sig@(4)) - 1 * (sig@(3)) - 126 * (sig@(2)) - 261 * (sig@(1)) - 406 * (sig@(0))) / 9889;

process(L, R) = left_out, right_out
with {
// JSFX HPF: y[n] = a*(y[n-1] + x[n] - x[n-1])
// Implemented as: u = (x - x@1); y = a*u + a*y@1
hpf_jsfx(a) = (_ <: (_, _@1) : -) : *(a) : (+ ~ *(a));

hpL = L : hpf_jsfx(a);
hpR = R : hpf_jsfx(a);





  ehf = max(abs(hpL), abs(hpR));
  env = ehf : (max ~ *(env_rel));
  baseFollower(a) = *(a) : (+ ~ (*(1.0 - a)));
  base = env : baseFollower(base_a);

  ratio = env / (base + eps);

  xC_L = L@(15);
  xC_R = R@(15);

  small_L = ba.selectn(3, mode, sg11_pred(L), sg15_pred(L), sg21_pred(L));
  small_R = ba.selectn(3, mode, sg11_pred(R), sg15_pred(R), sg21_pred(R));
  large_L = ba.selectn(3, mode, sg15_pred(L), sg21_pred(L), sg31_pred(L));
  large_R = ba.selectn(3, mode, sg15_pred(R), sg21_pred(R), sg31_pred(R));

  eA = max(abs(xC_L - small_L), abs(xC_R - small_R)) / (max(abs(small_L), abs(small_R)) + 1e-6);
  eB = max(abs(xC_L - large_L), abs(xC_R - large_R)) / (max(abs(large_L), abs(large_R)) + 1e-6);

  useA = eA <= eB;
  pred_L = ba.if(useA, small_L, large_L);
  pred_R = ba.if(useA, small_R, large_R);
  e_norm = ba.if(useA, eA, eB);

  trig = (ratio > ratio_thr) * (e_norm > err_thr);
  T = 1e-3;                         // threshold for “active”
  relHold = exp(log(T) / (holdN + eps));
  holdEnvFollower(rel) = max ~ (*(rel));
  holdEnv = trig : holdEnvFollower(relHold);
  active = holdEnv > T;


  range = err_thr * 3;
  mix_base = ba.if(active, clamp((e_norm - err_thr) / (range + eps), 0, 1), 0);
  mix = mix_base * mix_max;

  outL = xC_L * (1 - mix) + pred_L * mix;
  outR = xC_R * (1 - mix) + pred_R * mix;

  dL = outL - xC_L;
  dR = outR - xC_R;

  left_out  = ba.if(monitor, dL, outL);
  right_out = ba.if(monitor, dR, outR);
};