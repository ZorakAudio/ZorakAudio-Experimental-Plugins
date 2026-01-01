import("stdfaust.lib");

declare name "Reverb Tail Tamer (Wet 1/2, Ref In 5/6)";
declare version "3.0-norec";

// ------------------------- UI -------------------------
amount_dB = hslider("Amount (max duck dB)[unit:dB]", 12, 0, 24, 0.1);
sens_pct  = hslider("Sensitivity (%)",               50, 0, 100, 1);
rel_ms_ui = hslider("Release (ms)[unit:ms]",         350, 50, 1200, 1);

// --------------------- helpers ------------------------
eps = 1e-12;

clamp(x, lo, hi) = max(lo, min(hi, x));
smoothstep01(x) = x1*x1*(3 - 2*x1) with { x1 = clamp(x, 0, 1); };

db2lin(db) = pow(10, db/20);
lin2db(x)  = 20*log10(max(x, 1e-30));

sel(c, t, f) = ba.if(c, t, f);

ms2pole(ms) = exp(-1.0 / (ma.SR * (ms/1000.0))); // 0..1 pole for si.smooth
// si.smooth(pole) : exponential smoothing with controllable pole :contentReference[oaicite:1]{index=1}

// -------------------- constants -----------------------
atk_ms     = 12.0;
hold_ms    = 80.0;   // becomes "hold-ish" smoothing, not a countdown
rms_ms     = 35.0;

tgt_ms     = 25.0;
rel_in_ms  = 70.0;

dry_on_db  = -50.0;
ref_off_db = -60.0;

floor_db   = -80.0;
floor_lin  = db2lin(floor_db);

// ----------------------- core -------------------------
// 6-in / 6-out
tamer6(wetL, wetR, ch3, ch4, refL, refR) = wetL*g, wetR*g, ch3, ch4, refL, refR
with {
  sens       = sens_pct / 100.0;
  maxduck_dB = amount_dB;
  rel_ms     = rel_ms_ui;

  // Sensitivity mapping (UNCHANGED)
  thr_db  = 18.0 - sens*21.0;     // 18 .. -3 dB
  ratio   = 1.2 + sens*3.0;       // 1.2 .. 4.2
  knee_db = 10.0 - sens*6.0;      // ~10 .. ~4 dB

  // Tail ramp time follows Release
  grace_ms = clamp(rel_ms*0.25, 60.0, 200.0);

  // poles (for library smoother)
  pole_rms    = ms2pole(rms_ms);
  pole_tgt    = ms2pole(tgt_ms);
  pole_grace  = ms2pole(grace_ms);
  pole_hold   = ms2pole(hold_ms);
  pole_atk    = ms2pole(atk_ms);
  pole_rel    = ms2pole(rel_ms);
  pole_rel_in = ms2pole(rel_in_ms);

  dry_on_lin  = db2lin(dry_on_db);
  ref_off_lin = db2lin(ref_off_db);

  // detector power (stereo)
  wet_p = 0.5*(wetL*wetL + wetR*wetR);
  ref_p = 0.5*(refL*refL + refR*refR);

  // EMA-of-power using library smoother (no user recursion)
  wet_env2 = wet_p : si.smooth(pole_rms);
  ref_env2 = ref_p : si.smooth(pole_rms);

  Ey = max(sqrt(max(wet_env2, 0.0)), floor_lin);
  Ex = max(sqrt(max(ref_env2, 0.0)), floor_lin);

  dryA = (Ex > dry_on_lin);
  offA = (Ex <= ref_off_lin);

  // Tail ramp-in after ref goes inactive:
  // offA_s rises slowly when offA=1, instantly goes ~0 when offA=0 (via selection)
  offA_s = offA : si.smooth(pole_grace);
  tail_w = (1.0 - offA) + offA * smoothstep01(offA_s);

  // ratio-of-return in dB
  rdB = lin2db((Ey + eps) / (Ex + eps));

  // soft-knee overage
  over = rdB - thr_db;
  over_eff =
    sel(over <= 0.0,
        0.0,
        over * smoothstep01(clamp(over / max(knee_db, 0.001), 0.0, 1.0)));

  // target GR (positive dB), then tail weighting
  tgt0 = sel(over_eff > 0.0, min(maxduck_dB, over_eff*ratio), 0.0);
  tgt1 = tgt0 * tail_w;

  // smooth target (prevents stepping)
  tgt_db = tgt1 : si.smooth(pole_tgt);

  // --- JSFX-like GR behavior (stateful) using library AR follower ---
// This restores the “alive” motion when ref is muted.

// Smooth dry activity a little so switching release speeds doesn't click
dryA_s  = dryA : si.smooth(ms2pole(10.0));   // 10ms-ish switch smoothing

// Hold-ish pinning on the *target* (not on GR) but closer to JSFX than before:
// - only meaningful when ref is NOT active (dryA=0)
// - cancels immediately when ref returns (dryA=1)
tgt_hold = max(tgt_db, (tgt_db : si.smooth(ms2pole(hold_ms))));
tgt_pin  = (1.0 - dryA) * tgt_hold + dryA * tgt_db;

// Two AR followers: normal release vs faster release when ref is present
att_s    = atk_ms / 1000.0;
rel_s    = rel_ms / 1000.0;
rel_in_s = rel_in_ms / 1000.0;

gr_norm  = tgt_pin : an.amp_follower_ar(att_s, rel_s);
gr_fast  = tgt_pin : an.amp_follower_ar(att_s, rel_in_s);

// Crossfade between them based on ref activity
gr_db    = (1.0 - dryA_s)*gr_norm + dryA_s*gr_fast;


  g = db2lin(-gr_db);
};

process = tamer6;
