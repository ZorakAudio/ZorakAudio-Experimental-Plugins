// Pure FAUST with a shared detector: both channels receive the same gain.
// Threshold uses the library's abs(left)+abs(right) detector convention.
import("stdfaust.lib");
ratio = hslider("Ratio",3,1,12,0.1) : si.smoo;
threshold = hslider("Threshold [unit:dB]",-18,-48,0,0.1) : si.smoo;
attack = hslider("Attack [unit:ms]",10,1,100,1)*0.001;
release = hslider("Release [unit:ms]",150,20,1000,1)*0.001;
makeup = hslider("Makeup [unit:dB]",0,0,18,0.1) : ba.db2linear : si.smoo;
// A continuous 0..1 Wet control remains a slider, not a button.
mix = hslider("Wet",1,0,1,0.01) : si.smoo;
dry = checkbox("Bypass") : si.smoo;
// Both outputs use this one shared stereo gain computer.
gain(l,r) = co.compression_gain_mono(ratio,threshold,attack,release,abs(l)+abs(r));
blend(x,g) = x*((1-mix*(1-dry)) + mix*(1-dry)*g*makeup);
process(l,r) = blend(l,g),blend(r,g) with { g=gain(l,r); };
