// Pure FAUST: no JSFX header or section markers.
// The editor generates controls from these declarations automatically.
import("stdfaust.lib");
cutoff = hslider("Cutoff [unit:Hz]",1800,40,18000,1) : si.smoo;
q = hslider("Resonance (Q)",0.707,0.5,4,0.01) : si.smoo;
wet = hslider("Wet",1,0,1,0.01) : si.smoo;
level = hslider("Output [unit:dB]",0,-18,6,0.1) : ba.db2linear : si.smoo;
// Clamp the cutoff below Nyquist; resonant settings can increase peaks.
filter = fi.resonlp(min(cutoff,0.45*ma.SR),q,1);
channel(x) = level*((1-wet)*x + wet*filter(x));
process = channel,channel;
