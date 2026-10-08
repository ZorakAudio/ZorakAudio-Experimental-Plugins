#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
namespace jsfx_input {
inline int packCC(const char* value){uint32_t result=0;while(*value)result=(result<<8)|(uint8_t)*value++;return (int)result;}
    inline int keyPressToJsfx (const juce::KeyPress& key)
    {
        const int kc = key.getKeyCode();

        if (kc == juce::KeyPress::upKey)       return packCC ("up");
        if (kc == juce::KeyPress::downKey)     return packCC ("down");
        if (kc == juce::KeyPress::leftKey)     return packCC ("left");
        if (kc == juce::KeyPress::rightKey)    return packCC ("rght");
        if (kc == juce::KeyPress::homeKey)     return packCC ("home");
        if (kc == juce::KeyPress::endKey)      return packCC ("end");
        if (kc == juce::KeyPress::pageUpKey)   return packCC ("pgup");
        if (kc == juce::KeyPress::pageDownKey) return packCC ("pgdn");
        if (kc == juce::KeyPress::insertKey)   return packCC ("ins");
        if (kc == juce::KeyPress::deleteKey)   return packCC ("del");

        if (kc == juce::KeyPress::escapeKey)   return 27;
        if (kc == juce::KeyPress::returnKey)   return 13;
        if (kc == juce::KeyPress::spaceKey)    return 32;

        if (kc >= juce::KeyPress::F1Key && kc <= juce::KeyPress::F12Key)
        {
            const int f = (kc - juce::KeyPress::F1Key) + 1;
            if (f < 10)
            {
                const char s[3] = { 'f', (char) ('0' + f), 0 };
                return packCC (s);
            }

            const char s[4] = { 'f', (char) ('0' + (f / 10)), (char) ('0' + (f % 10)), 0 };
            return packCC (s);
        }

        if (kc == juce::KeyPress::backspaceKey) return 8;
        if (kc == juce::KeyPress::tabKey) return 9;
        const auto ch = key.getTextCharacter();
       #if JUCE_MAC
        const bool control = key.getModifiers().isCommandDown();
       #else
        const bool control = key.getModifiers().isCtrlDown();
       #endif
        const bool alt = key.getModifiers().isAltDown();
        const int upper = (kc >= 'a' && kc <= 'z') ? kc - ('a' - 'A') : kc;
        if (upper >= 'A' && upper <= 'Z')
        {
            if (control) return upper - 'A' + 1 + (alt ? 256 : 0);
            if (alt) return upper + 256;
        }
        if (ch > 127 && ch <= 0x10ffff && !(ch >= 0xd800 && ch <= 0xdfff))
            return (int) (((uint32_t) 'u' << 24) | (uint32_t) ch);
        if (ch > 0 && ch <= 127) return (int) ch + (alt ? 256 : 0);
        return 0;
    }


}
