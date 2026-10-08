#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <type_traits>

namespace jit_controls {
// Menu callbacks belong to the widget, so replacing a compiled schema or
// closing the editor cannot leave a callback pointing at a retired control.
template <class Base>
class DefaultControl final : public Base {
public:
    juce::String defaultDescription;
    std::function<void()> onReset;

    juce::PopupMenu resetMenu() const {
        juce::PopupMenu menu;
        menu.addItem(1, "Reset to default (" + defaultDescription + ")", bool(onReset));
        return menu;
    }
    void performReset() { if (onReset) onReset(); }
    void mouseDown(const juce::MouseEvent& event) override {
        if constexpr (std::is_same_v<Base, juce::Slider>) {
            if (!event.mods.isPopupMenu() && event.getNumberOfClicks() > 1) return;
        }
        if (!event.mods.isPopupMenu()) { Base::mouseDown(event); return; }
        if (!this->isEnabled()) return;
        auto menu = resetMenu();
        menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this).withMousePosition(),
            [safe = juce::Component::SafePointer<DefaultControl>(this)](int result) {
                if (safe && result == 1) safe->performReset();
            });
    }
    void mouseDoubleClick(const juce::MouseEvent& event) override {
        if constexpr (std::is_same_v<Base, juce::Slider>) {
            if (this->isEnabled() && !event.mods.isPopupMenu()) performReset();
        } else {
            Base::mouseDoubleClick(event);
        }
    }
};
using DefaultSlider = DefaultControl<juce::Slider>;
using DefaultChoice = DefaultControl<juce::ComboBox>;
using DefaultLabel = DefaultControl<juce::Label>;
using DefaultToggle = DefaultControl<juce::ToggleButton>;
using DefaultButton = DefaultControl<juce::TextButton>;

class DefaultTextEditor final : public juce::TextEditor {
public:
    static constexpr int resetItem = 0x300000;
    juce::String defaultDescription;
    std::function<void()> onReset;
    void addPopupMenuItems(juce::PopupMenu& menu, const juce::MouseEvent* event) override {
        TextEditor::addPopupMenuItems(menu, event);
        menu.addSeparator();
        menu.addItem(resetItem, "Reset to default (" + defaultDescription + ")", bool(onReset));
    }
    void performPopupMenuAction(int item) override {
        if (item == resetItem) { if (onReset) onReset(); }
        else TextEditor::performPopupMenuAction(item);
    }
};
} // namespace jit_controls
