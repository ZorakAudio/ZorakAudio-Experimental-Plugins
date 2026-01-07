#pragma once
#include <cstdint>

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif

struct Meta {
    virtual ~Meta() = default;
    virtual void declare(const char* key, const char* value) = 0;
};

struct UI {
    virtual ~UI() = default;

    virtual void openTabBox(const char* label) = 0;
    virtual void openHorizontalBox(const char* label) = 0;
    virtual void openVerticalBox(const char* label) = 0;
    virtual void closeBox() = 0;

    virtual void addButton(const char* label, FAUSTFLOAT* zone) = 0;
    virtual void addCheckButton(const char* label, FAUSTFLOAT* zone) = 0;

    virtual void addVerticalSlider(const char* label, FAUSTFLOAT* zone,
                                   FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) = 0;
    virtual void addHorizontalSlider(const char* label, FAUSTFLOAT* zone,
                                     FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) = 0;
    virtual void addNumEntry(const char* label, FAUSTFLOAT* zone,
                             FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) = 0;

    virtual void addHorizontalBargraph(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min, FAUSTFLOAT max) = 0;
    virtual void addVerticalBargraph(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min, FAUSTFLOAT max) = 0;

    // Optional metadata hook: many generated DSPs call this
    virtual void declare(FAUSTFLOAT* zone, const char* key, const char* value) = 0;
};

// Soundfile support (we don't use it, but generated code references it in UI interface)
struct Soundfile {};

class dsp {
public:
    virtual ~dsp() = default;

    virtual int getNumInputs() = 0;
    virtual int getNumOutputs() = 0;

    virtual void buildUserInterface(UI* ui) = 0;
    virtual void metadata(Meta* m) = 0;

    virtual void init(int sample_rate) = 0;
    virtual void compute(int count, FAUSTFLOAT** inputs, FAUSTFLOAT** outputs) = 0;
};
