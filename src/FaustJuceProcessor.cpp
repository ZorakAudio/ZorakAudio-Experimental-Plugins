#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include "faust_support_min.h"
#include "FaustDSP.h"

namespace
{
    // Remove Faust-style metadata like: "Gain [unit:dB][scale:log]"
    static juce::String stripLabelAndCollectMeta (const juce::String& in,
                                                  std::map<std::string, std::string>* metaOut)
    {
        juce::String label = in;
        int start = 0;

        while ((start = label.indexOfChar ('[')) >= 0)
        {
            const int end = label.indexOfChar (start, ']');
            if (end < 0) break;

            const auto inside = label.substring (start + 1, end).trim();
            const int colon = inside.indexOfChar (':');

            if (metaOut != nullptr && colon > 0)
            {
                const auto k = inside.substring (0, colon).trim().toStdString();
                const auto v = inside.substring (colon + 1).trim().toStdString();
                (*metaOut)[k] = v;
            }

            label = (label.substring (0, start) + label.substring (end + 1)).trim();
        }

        return label.trim();
    }

    static juce::String sanitizeId (const juce::String& s)
    {
        juce::String out;
        for (auto c : s)
        {
            if (juce::CharacterFunctions::isLetterOrDigit (c)) out << c;
            else if (c == ' ' || c == '-' || c == '_') out << '_';
        }
        if (out.isEmpty()) out = "param";
        return out;
    }

    struct FaustParam
    {
        juce::String id;
        juce::String name;
        FAUSTFLOAT* zone = nullptr;

        float init = 0.0f;
        float min  = 0.0f;
        float max  = 1.0f;
        float step = 0.0f;

        bool isDiscrete01 = false;
        std::map<std::string, std::string> meta;
    };

    struct ParamCollectorUI : public UI
    {
        std::vector<juce::String> groupStack;
        std::vector<FaustParam> params;

        // Faust UI layout
        void openTabBox (const char* label) override         { pushGroup (label); }
        void openHorizontalBox (const char* label) override  { pushGroup (label); }
        void openVerticalBox (const char* label) override    { pushGroup (label); }
        void closeBox() override                             { popGroup(); }

        // Active widgets
        void addButton (const char* label, FAUSTFLOAT* zone) override
        {
            addParam (label, zone, 0.0f, 0.0f, 1.0f, 1.0f, true);
        }

        void addCheckButton (const char* label, FAUSTFLOAT* zone) override
        {
            addParam (label, zone, 0.0f, 0.0f, 1.0f, 1.0f, true);
        }

        void addVerticalSlider (const char* label, FAUSTFLOAT* zone,
                                FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false);
        }

        void addHorizontalSlider (const char* label, FAUSTFLOAT* zone,
                                  FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false);
        }

        void addNumEntry (const char* label, FAUSTFLOAT* zone,
                          FAUSTFLOAT init, FAUSTFLOAT min, FAUSTFLOAT max, FAUSTFLOAT step) override
        {
            addParam (label, zone, (float) init, (float) min, (float) max, (float) step, false);
        }

        // Passive widgets (ignored for parameters)
        void addHorizontalBargraph (const char*, FAUSTFLOAT*, FAUSTFLOAT, FAUSTFLOAT) override {}
        void addVerticalBargraph   (const char*, FAUSTFLOAT*, FAUSTFLOAT, FAUSTFLOAT) override {}

        // Soundfiles (ignored)
        // void addSoundfile (const char*, const char*, Soundfile**) override {}

        // Metadata hook (optional; labels already carry most metadata)
        void declare (FAUSTFLOAT*, const char*, const char*) override {}

    private:
        void pushGroup (const char* raw)
        {
            std::map<std::string, std::string> meta;
            const auto clean = stripLabelAndCollectMeta (raw, &meta);
            groupStack.push_back (clean);
        }

        void popGroup()
        {
            if (!groupStack.empty())
                groupStack.pop_back();
        }

        juce::String makeId (const juce::String& leaf) const
        {
            juce::StringArray parts;
            for (auto& g : groupStack)
                if (g.isNotEmpty())
                    parts.add (sanitizeId (g));

            parts.add (sanitizeId (leaf));
            return parts.joinIntoString ("__");
        }

        void addParam (const char* rawLabel, FAUSTFLOAT* zone,
                       float init, float min, float max, float step,
                       bool discrete01)
        {
            FaustParam p;
            p.meta.clear();

            p.name = stripLabelAndCollectMeta (rawLabel, &p.meta);
            p.id   = makeId (p.name);

            p.zone = zone;
            p.init = init;
            p.min  = min;
            p.max  = max;
            p.step = step;
            p.isDiscrete01 = discrete01;

            params.push_back (p);
        }
    };
}

class FaustJuceProcessor final : public juce::AudioProcessor
{
public:
    FaustJuceProcessor() : juce::AudioProcessor (makeBusesFromFaust())
    {
        dsp = std::make_unique<mydsp>();

        // Collect parameters from Faust UI
        dsp->buildUserInterface (&ui);
        faustParams = ui.params;

        juce::AudioProcessorValueTreeState::ParameterLayout layout;

        for (const auto& p : faustParams)
        {
            const auto range = juce::NormalisableRange<float> (p.min, p.max, p.step);
            layout.add (std::make_unique<juce::AudioParameterFloat> (p.id, p.name, range, p.init));
        }

        apvts = std::make_unique<juce::AudioProcessorValueTreeState> (*this, nullptr, "PARAMS", std::move (layout));
    }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void prepareToPlay (double sampleRate, int /*samplesPerBlock*/) override
    {
        dsp->init ((int) sampleRate);

        inputPtrs.resize ((size_t) dsp->getNumInputs());
        outputPtrs.resize ((size_t) dsp->getNumOutputs());
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
{
    const int ins  = dsp->getNumInputs();
    const int outs = dsp->getNumOutputs();

    const auto inSet  = layouts.getMainInputChannelSet();
    const auto outSet = layouts.getMainOutputChannelSet();

    if (ins > 0 && inSet.size() != ins)  return false;
    if (outs > 0 && outSet.size() != outs) return false;

    // If you want “must match in/out” for typical effects, keep this:
    if (ins > 0 && outs > 0 && inSet.size() != outSet.size())
        return false;

    return true;
}


    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;

        // Push JUCE param values into Faust zones
        for (auto& p : faustParams)
        {
            if (auto* v = apvts->getRawParameterValue (p.id))
                *p.zone = (FAUSTFLOAT) v->load();
        }

        const int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < dsp->getNumInputs(); ++ch)
            inputPtrs[(size_t) ch] = (FAUSTFLOAT*) buffer.getReadPointer (ch);

        for (int ch = 0; ch < dsp->getNumOutputs(); ++ch)
            outputPtrs[(size_t) ch] = (FAUSTFLOAT*) buffer.getWritePointer (ch);

        dsp->compute (numSamples, inputPtrs.data(), outputPtrs.data());
    }

    juce::AudioProcessorEditor* createEditor() override
    {
        return new juce::GenericAudioProcessorEditor (*this);
    }

    bool hasEditor() const override { return true; }

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        auto state = apvts->copyState();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));
        if (xml && xml->hasTagName (apvts->state.getType()))
            apvts->replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    std::unique_ptr<mydsp> dsp;
    ParamCollectorUI ui;

    std::unique_ptr<juce::AudioProcessorValueTreeState> apvts;
    std::vector<FaustParam> faustParams;

    std::vector<FAUSTFLOAT*> inputPtrs;
    std::vector<FAUSTFLOAT*> outputPtrs;

    static BusesProperties makeBusesFromFaust()
    {
        mydsp d;

        const int ins  = d.getNumInputs();
        const int outs = d.getNumOutputs();

        // Use discrete sets for >2 channels so hosts can wire arbitrary multichannel.
        auto setFor = [] (int ch) -> juce::AudioChannelSet
        {
            if (ch <= 0) return juce::AudioChannelSet::disabled();
            if (ch == 1) return juce::AudioChannelSet::mono();
            if (ch == 2) return juce::AudioChannelSet::stereo();
            return juce::AudioChannelSet::discreteChannels (ch);
        };

        BusesProperties props;
        if (ins > 0)  props = props.withInput  ("Input",  setFor(ins),  true);
        if (outs > 0) props = props.withOutput ("Output", setFor(outs), true);

        return props;
    }

};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new FaustJuceProcessor();
}
