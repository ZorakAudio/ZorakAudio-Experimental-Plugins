#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <cmath>
#include <stdexcept>
inline juce::StringArray makeRuntimeBank(const juce::File& directory,int count=3){
    if(directory.createDirectory().failed())throw std::runtime_error("Create synthetic bank");
    juce::StringArray paths;
    for(int item=0;item<count;++item){
        auto file=directory.getChildFile("fixture-"+juce::String(item)+".wav");auto stream=file.createOutputStream();
        if(!stream)throw std::runtime_error("Open synthetic WAV");stream->setPosition(0);stream->truncate();
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(),48000,2,24,{},0));
        if(!writer)throw std::runtime_error("Encode synthetic WAV");stream.release();juce::AudioBuffer<float> audio(2,48000);
        for(int c=0;c<2;++c)for(int f=0;f<48000;++f){const double t=f/48000.;audio.setSample(c,f,float(.3*std::min(1.,t/.003)*std::exp(-t*(3+item))*std::sin(2*juce::MathConstants<double>::pi*((170+item*110)*t+80*t*t)+c*.03)));}
        if(!writer->writeFromAudioSampleBuffer(audio,0,48000))throw std::runtime_error("Write synthetic WAV");paths.add(file.getFullPathName());
    }
    return paths;
}
