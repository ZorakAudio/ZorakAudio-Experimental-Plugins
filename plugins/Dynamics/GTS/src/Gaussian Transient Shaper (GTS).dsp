import("stdfaust.lib");

// ------------------------------------------------------------
// Metadata
// ------------------------------------------------------------
declare name      "Gaussian Transient Shaper";
declare author    "Zorak Audio";
declare version   "1.0";
declare latency_frames "128";

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------
GAUSS_RADIUS = 128;                  // half kernel length
GAUSS_LEN    = 2*GAUSS_RADIUS + 1;   // total taps (257)


// ------------------------------------------------------------
// True Gaussian FIR kernel (time-domain)
// h[n] ∝ exp(-0.5 * (n/sigma)^2), truncated to ±GAUSS_RADIUS
// Symmetric, linear phase, normalized to DC gain = 1
// ------------------------------------------------------------
gaussKernel(sigmaSamples) = par(k, GAUSS_LEN, coeff(k))
with {
    // discrete Gaussian at integer offset i ≥ 0
    g(i)   = exp(-0.5 * pow(float(i) / sigmaSamples, 2.0));

    // normalization for symmetric kernel over [-R..+R]
    g0      = g(0);
    sumRest = sum(i, GAUSS_RADIUS, g(i+1));   // g(1) + ... + g(R)
    norm    = 1.0 / (g0 + 2.0 * sumRest + 1e-20);

    // index k runs 0..GAUSS_LEN-1, center at GAUSS_RADIUS
    offset(k) = abs(k - GAUSS_RADIUS);
    coeff(k)  = norm * g(offset(k));
};

// FIR Gaussian blur with runtime sigma (in samples)
gaussianBlur(sigmaSamples, x) = x : fi.fir(gaussKernel(sigmaSamples));

// ------------------------------------------------------------
// Core transient shaper
// Sustain  = Gaussian blur
// Attack   = dry(aligned) - Sustain   (inversion of blur)
// ------------------------------------------------------------
gaussTransient(x) = y
with {
    
    // ---------------- UI ----------------
    // Sigma in milliseconds (Gaussian std‑dev, not window length)
    // Rough useful range for transient work: sub‑ms to ~8 ms
    sigmaMs = vgroup("Control", hslider("Gaussian Sigma (σ) [unit:ms]",
                      2.0, 0.1, 8.0, 0.01) : si.smoo);

    attackDB   = vgroup("Control", hslider("Attack Gain [unit:dB]",
                         0, -12, 12, 0.1) : si.smoo);
    sustainDB  = vgroup("Control", hslider("Sustain Gain [unit:dB]",
                         0, -12, 12, 0.1) : si.smoo);

    mix        = vgroup("Gain", hslider("Mix [style:knob]",
                         1.0, 0.0, 1.0, 0.01) : si.smoo);

    outGain    = vgroup("Gain",hslider("Output Gain [unit:dB]",
                         0, -12, 12, 0.1) : ba.db2linear : si.smoo);

    // ------------- Gaussian params -------------
    // sigma in *samples*. Clamp to avoid division problems.
    sigmaSamplesRaw = sigmaMs * ma.SR * 0.001;
    sigmaSamples    = max(0.25, sigmaSamplesRaw);

    // ------------- Signal flow -------------
    xIn      = x;

    // Gaussian sustain (linear-phase FIR, latency = GAUSS_RADIUS)
    sustain  = gaussianBlur(sigmaSamples, xIn);

    // Delay dry path by GAUSS_RADIUS to match FIR group delay
    // de.delay(maxDelay, currentDelay)
    xAligned = xIn : de.delay(GAUSS_RADIUS, GAUSS_RADIUS);

    // Attack = inversion of Gaussian blur
    attack   = xAligned - sustain;

    aGain    = ba.db2linear(attackDB);
    sGain    = ba.db2linear(sustainDB);

    shaped   = aGain*attack + sGain*sustain;

    // Wet/dry blend, then output gain
    y        = ((mix*shaped) + ((1.0 - mix)*xAligned)) * outGain;
};

// ------------------------------------------------------------
// Stereo wrapper
// ------------------------------------------------------------
process = vgroup("Gaussian Transient Shaper",
                 gaussTransient, gaussTransient);
