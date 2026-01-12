declare name        "Designed Panning Topology (DPT) — Headphone Psycho Panner (SAFE)";
declare description "SAFE controls: Azimuth + Amount + Output. Internally: equal-power pan + far-ear ITD + far-ear shadow + conservative crossfeed.";
declare latency_frames "0";
import("stdfaust.lib");

// -------------------------
// Utilities
// -------------------------
SR = ma.SR;
PI = ma.PI;

clamp(lo, hi, x) = min(hi, max(lo, x));
clamp01(x)        = clamp(0.0, 1.0, x);

smooth01(t) = u*u*(3.0 - 2.0*u) with { u = clamp01(t); };

db2lin(db) = pow(10.0, db / 20.0);

// One-pole lowpass coefficient from cutoff (Hz), JSFX-style:
// y[n] = (1-a)*x[n] + a*y[n-1]
alphaLP(fc) = a with {
  fcC = clamp(40.0, 0.49*SR, fc);
  a0  = exp(-2.0*PI*fcC/SR);
  a   = clamp(0.0, 0.999999, a0);
};

onepole(a) = *(1.0 - a) : (+ ~ *(a));

// sel expected 0..1
mix01(sel, a, b) = a*(1.0-sel) + b*sel;

clip8(x) = clamp(-8.0, 8.0, x);

// -------------------------
// UI (SAFE controls only)
// -------------------------
azimuth = hslider("Azimuth[unit:%][style:knob]", 0, -100, 100, 1);
amount  = hslider("Amount[unit:%][style:knob]", 70, 0, 100, 1);
outDb   = hslider("Output (dB)[unit:dB][style:knob]", 0, -12, 12, 0.1);

// -------------------------
// DPT Core (Headphones-only, SAFE controls)
// -------------------------
dpt(inL, inR) = outL, outR with {

  pan = clamp(-1.0, 1.0, azimuth / 100.0);
  amt = smooth01(amount / 100.0);

  outGain = db2lin(clamp(-12.0, 12.0, outDb));

  // Mono panning source (predictable)
  x = 0.5 * (inL + inR);

  // Equal-power base gains (0..pi/2)
  theta = (pan + 1.0) * (PI/4.0);
  gL0   = cos(theta);
  gR0   = sin(theta);

  // Blend from center to full pan using Amount
  c0 = 0.70710678; // ~1/sqrt(2)
  gL = c0 + amt*(gL0 - c0);
  gR = c0 + amt*(gR0 - c0);

  absP = abs(pan);

  // -------------------------
  // ITD model (far ear): max ~0.63 ms @ 90 deg, clamped to 32 samples
  // Scales with Amount and pan magnitude
  // -------------------------
  itdSec  = 0.00063 * sin(absP * (PI/2.0)) * amt;
  itdSamp = min(32, int(itdSec*SR + 0.5));

  // -------------------------
  // Shadow (fixed SAFE strength curve driven by Amount)
  // Keep gentle: cutoff stays >= ~900 Hz, <= 20 kHz
  // -------------------------
  // Effective shadow amount grows with angle and Amount, but is capped.
  shEff = clamp01(0.75 * amt); // SAFE cap (never full "muffle")

  fc_sh_raw = 20000.0 / (1.0 + 8.0*shEff*pow(absP, 1.2)*amt);
  fc_sh     = clamp(900.0, 20000.0, fc_sh_raw);
  a_sh      = alphaLP(fc_sh);

  // -------------------------
  // Crossfeed (fixed SAFE amount driven by Amount)
  // Conservative: max gain ~= 0.15 (about -16.5 dB), only when Amount high
  // -------------------------
  xfGain = 0.15 * amt;

  xfDelRaw = (0.00025 + 0.00025*absP) * SR;
  xfDel    = min(48, int(xfDelRaw + 0.5));

  // Conservative crossfeed LP: mid-ish warmth, not dull
  fc_xf_raw = 1200.0 + 800.0*(1.0 - absP);
  fc_xf     = clamp(500.0, 3500.0, fc_xf_raw);
  a_xf      = alphaLP(fc_xf);

  // Small bounded delays (safe)
  xd  = x : de.delay(64, itdSamp);   // far-ear ITD sample
  xxf = x : de.delay(64, xfDel);     // crossfeed tap

  // One-pole LPs
  shOut = xd  : onepole(a_sh);       // shadowed far-ear signal
  xfOut = xxf : onepole(a_xf);       // crossfeed filtered signal

  // Decide which ear is far (pan>0 => left is far, right is near)
  farIsLeft = (pan > 0.0);

  // Near ear = undelayed x with base gain
  nearL = gL * x;
  nearR = gR * x;

  // Far ear = delayed+shadowed with base gain + crossfeed contribution
  farL  = gL * shOut + xfGain * xfOut;
  farR  = gR * shOut + xfGain * xfOut;

  // Route: if farIsLeft=1 => L=far, R=near; else L=near, R=far
  oL = mix01(farIsLeft, nearL, farL);
  oR = mix01(farIsLeft, farR,  nearR);

  // Output gain + safety clamp
  outL = clip8(oL * outGain);
  outR = clip8(oR * outGain);
};

process = dpt;
