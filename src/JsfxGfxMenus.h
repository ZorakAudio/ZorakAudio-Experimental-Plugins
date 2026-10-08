#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "JsfxGfxMenuPort.h"
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
class GfxMenuOverlay final : public juce::Component
{
public:
    struct Item
    {
        juce::String text;
        int resultId = 0;
        bool separator = false;
        bool disabled = false;
        bool checked = false;
        std::vector<Item> children;
    };

    std::function<void (int)> onFinished;

    GfxMenuOverlay()
    {
        setVisible (false);
        setOpaque (false);
        setInterceptsMouseClicks (true, true);
        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);
    }

    bool isMenuShowing() const noexcept { return menuOpen; }

    void openMenu (const juce::String& desc, juce::Point<int> anchorPos)
    {
        std::vector<Item> parsed;
        int nextId = 1;
        const auto utf8 = desc.toStdString();
        const char* cursor = utf8.c_str();
        parseRecursive (cursor, 0, nextId, parsed);

        rootItems = std::move (parsed);
        if (rootItems.empty())
        {
            finishNow (0, false);
            return;
        }

        anchor = anchorPos;
        menuOpen = true;
        finishScheduled = false;
        scheduledResult = 0;
        scheduledNotify = false;
        openPath.clear();
        highlightIndices.clear();
        highlightIndices.push_back (firstSelectableIndex (rootItems));

        if (auto* parent = getParentComponent())
            setBounds (parent->getLocalBounds());
        else
            setBounds (getBounds());

        rebuildLayout();
        setVisible (true);
        toFront (false);
        grabKeyboardFocus();
        repaint();
    }

    void forceCloseSilently()
    {
        finishScheduled = false;
        scheduledResult = 0;
        scheduledNotify = false;
        finishNow (0, false);
    }

    void paint (juce::Graphics& g) override
    {
        if (! menuOpen)
            return;

        static constexpr int kShadowRadius = 6;
        const auto bg = juce::Colour::fromRGBA (28, 30, 34, 240);
        const auto border = juce::Colour::fromRGBA (180, 188, 198, 230);
        const auto textColour = juce::Colour::fromRGB (235, 238, 242);
        const auto disabledColour = juce::Colour::fromRGB (118, 124, 132);
        const auto highlight = juce::Colour::fromRGB (70, 92, 132);

        for (size_t level = 0; level < layoutCache.size(); ++level)
        {
            const auto& panel = layoutCache[level];

            juce::DropShadow (juce::Colours::black.withAlpha (0.45f), kShadowRadius, { 0, 2 })
                .drawForRectangle (g, panel.bounds);

            g.setColour (bg);
            g.fillRect (panel.bounds);

            g.setColour (border);
            g.drawRect (panel.bounds, 1);

            const auto* items = getItemsForLevel ((int) level);
            if (items == nullptr)
                continue;

            for (int i = 0; i < (int) items->size() && i < (int) panel.itemBounds.size(); ++i)
            {
                const auto& item = (*items)[(size_t) i];
                const auto row = panel.itemBounds[(size_t) i];

                if (item.separator)
                {
                    g.setColour (juce::Colours::white.withAlpha (0.14f));
                    g.drawLine ((float) row.getX() + 8.0f,
                                (float) row.getCentreY(),
                                (float) row.getRight() - 8.0f,
                                (float) row.getCentreY(),
                                1.0f);
                    continue;
                }

                const bool highlighted = isHighlighted ((int) level, i);
                if (highlighted && ! item.disabled)
                {
                    g.setColour (highlight);
                    g.fillRect (row.reduced (2, 1));
                }

                const int checkLeft = row.getX() + 8;
                const int textLeft  = row.getX() + 26;
                const int arrowRight = row.getRight() - 10;

                if (item.checked)
                {
                    g.setColour (item.disabled ? disabledColour : textColour);
                    juce::Path tick;
                    tick.startNewSubPath ((float) checkLeft,       (float) row.getCentreY());
                    tick.lineTo          ((float) checkLeft + 4.0f, (float) row.getCentreY() + 4.0f);
                    tick.lineTo          ((float) checkLeft + 10.0f,(float) row.getCentreY() - 4.0f);
                    g.strokePath (tick, juce::PathStrokeType (2.0f));
                }

                g.setColour (item.disabled ? disabledColour : textColour);
                g.setFont (menuFont);
                g.drawText (item.text,
                            juce::Rectangle<int> (textLeft, row.getY(), row.getWidth() - 40, row.getHeight()),
                            juce::Justification::centredLeft,
                            true);

                if (! item.children.empty())
                {
                    juce::Path arrow;
                    const float cx = (float) arrowRight;
                    const float cy = (float) row.getCentreY();
                    arrow.startNewSubPath (cx - 3.0f, cy - 5.0f);
                    arrow.lineTo          (cx + 3.0f, cy);
                    arrow.lineTo          (cx - 3.0f, cy + 5.0f);
                    arrow.closeSubPath();
                    g.fillPath (arrow);
                }
            }
        }
    }

    void resized() override
    {
        if (menuOpen)
            rebuildLayout();
    }

    void mouseMove (const juce::MouseEvent& e) override   { updateHover (e.getPosition()); }
    void mouseDrag (const juce::MouseEvent& e) override   { updateHover (e.getPosition()); }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (! menuOpen)
            return;

        // Do not close/activate on mouseDown(). If the overlay disappears before
        // mouseUp(), JUCE will deliver the release to the underlying @gfx view,
        // which can immediately retrigger the source control. Keep the overlay
        // alive until mouseUp so the full click gesture is consumed here.
        updateHover (e.getPosition());
    }

    void mouseUp (const juce::MouseEvent& e) override
    {
        if (! menuOpen)
            return;

        const auto pos = e.getPosition();

        int level = -1, index = -1;
        if (! hitTestPanels (pos, level, index))
        {
            scheduleFinish (0, true);
            return;
        }

        activateItem (level, index);
    }

    bool keyPressed (const juce::KeyPress& key) override
    {
        if (! menuOpen)
            return false;

        const int currentLevel = juce::jmax (0, (int) layoutCache.size() - 1);
        const auto* items = getItemsForLevel (currentLevel);
        if (items == nullptr || items->empty())
            return true;

        if (key == juce::KeyPress::escapeKey)
        {
            scheduleFinish (0, true);
            return true;
        }

        if (key == juce::KeyPress::upKey || key == juce::KeyPress::downKey)
        {
            const int dir = (key == juce::KeyPress::upKey ? -1 : 1);
            ensureHighlightSize (currentLevel + 1);

            int idx = highlightIndices[(size_t) currentLevel];
            idx = nextSelectableIndex (*items, idx, dir);
            highlightIndices[(size_t) currentLevel] = idx;

            if (idx >= 0 && idx < (int) items->size() && ! (*items)[(size_t) idx].children.empty())
            {
                ensureOpenPathSize (currentLevel + 1);
                openPath[(size_t) currentLevel] = idx;
                const auto* childItems = getItemsForLevel (currentLevel + 1);
                ensureHighlightSize (currentLevel + 2);
                highlightIndices[(size_t) (currentLevel + 1)] = childItems ? firstSelectableIndex (*childItems) : -1;
            }
            else
            {
                if ((int) openPath.size() > currentLevel)
                    openPath.resize ((size_t) currentLevel);
                if ((int) highlightIndices.size() > currentLevel + 1)
                    highlightIndices.resize ((size_t) currentLevel + 1);
            }

            rebuildLayout();
            repaint();
            return true;
        }

        if (key == juce::KeyPress::leftKey)
        {
            if ((int) openPath.size() > 0)
            {
                openPath.resize ((size_t) juce::jmax (0, currentLevel - 1));
                if ((int) highlightIndices.size() > juce::jmax (1, currentLevel))
                    highlightIndices.resize ((size_t) juce::jmax (1, currentLevel));
                rebuildLayout();
                repaint();
            }
            else
            {
                scheduleFinish (0, true);
            }
            return true;
        }

        if (key == juce::KeyPress::rightKey || key == juce::KeyPress::returnKey || key == juce::KeyPress::spaceKey)
        {
            const int idx = currentHighlight (currentLevel);
            activateItem (currentLevel, idx);
            return true;
        }

        return true;
    }

    void focusLost (juce::Component::FocusChangeType) override
    {
        if (menuOpen)
            scheduleFinish (0, true);
    }

private:
    struct PanelLayout
    {
        juce::Rectangle<int> bounds;
        std::vector<juce::Rectangle<int>> itemBounds;
    };

    static bool parseRecursive (const char*& cursor, int depth, int& nextId, std::vector<Item>& out)
    {
        if (cursor == nullptr || depth >= 8)
            return false;

        bool any = false;

        while (true)
        {
            const char* sep = std::strchr (cursor, '|');
            const size_t len = (sep != nullptr) ? (size_t) (sep - cursor) : std::strlen (cursor);

            std::string token (cursor, len);
            cursor += len;
            if (sep != nullptr)
                ++cursor;

            const char* q = token.c_str();
            bool done = false;
            bool disabled = false;
            bool checked = false;
            bool hasSubmenu = false;
            std::vector<Item> children;

            while (*q != '\0' && std::strchr (">#!<", *q) != nullptr)
            {
                if (*q == '>')
                    hasSubmenu = true;
                else if (*q == '#')
                    disabled = true;
                else if (*q == '!')
                    checked = true;
                else if (*q == '<')
                    done = true;
                ++q;
            }

            if (hasSubmenu)
                parseRecursive (cursor, depth + 1, nextId, children);

            if (*q != '\0')
            {
                Item item;
                item.text = juce::String::fromUTF8 (q).trim();
                item.disabled = disabled;
                item.checked = checked;
                item.children = std::move (children);
                item.resultId = item.children.empty() ? nextId++ : 0;

                if (! item.text.isEmpty())
                {
                    out.push_back (std::move (item));
                    any = true;
                }
            }
            else if (! hasSubmenu && ! done)
            {
                Item separator;
                separator.separator = true;
                out.push_back (std::move (separator));
                any = true;
            }

            if (sep == nullptr || done)
                break;
        }

        return any;
    }

    static int firstSelectableIndex (const std::vector<Item>& items)
    {
        for (int i = 0; i < (int) items.size(); ++i)
        {
            if (! items[(size_t) i].separator && ! items[(size_t) i].disabled)
                return i;
        }
        return -1;
    }

    static int nextSelectableIndex (const std::vector<Item>& items, int start, int dir)
    {
        if (items.empty())
            return -1;

        int idx = start;
        if (idx < 0 || idx >= (int) items.size())
            idx = (dir >= 0 ? -1 : (int) items.size());

        for (int count = 0; count < (int) items.size(); ++count)
        {
            idx += dir;
            if (idx < 0)
                idx = (int) items.size() - 1;
            if (idx >= (int) items.size())
                idx = 0;

            const auto& item = items[(size_t) idx];
            if (! item.separator && ! item.disabled)
                return idx;
        }

        return start;
    }

    const std::vector<Item>* getItemsForLevel (int level) const
    {
        if (level < 0)
            return nullptr;

        const std::vector<Item>* items = &rootItems;
        for (int l = 0; l < level; ++l)
        {
            if (l >= (int) openPath.size())
                return nullptr;

            const int idx = openPath[(size_t) l];
            if (idx < 0 || idx >= (int) items->size())
                return nullptr;

            items = &(*items)[(size_t) idx].children;
        }
        return items;
    }

    void ensureOpenPathSize (int size)
    {
        if ((int) openPath.size() < size)
            openPath.resize ((size_t) size, -1);
    }

    void ensureHighlightSize (int size)
    {
        if ((int) highlightIndices.size() < size)
            highlightIndices.resize ((size_t) size, -1);
    }

    int currentHighlight (int level) const
    {
        if (level < 0 || level >= (int) highlightIndices.size())
            return -1;
        return highlightIndices[(size_t) level];
    }

    bool isHighlighted (int level, int index) const
    {
        return currentHighlight (level) == index;
    }

    void rebuildLayout()
    {
        layoutCache.clear();
        if (! menuOpen)
            return;

        static constexpr int kBorder = 1;
        static constexpr int kRowHeight = 22;
        static constexpr int kSeparatorHeight = 8;
        static constexpr int kMinWidth = 120;
        static constexpr int kCheckWidth = 16;
        static constexpr int kArrowWidth = 14;
        static constexpr int kHorizPad = 12;

        const auto bounds = getLocalBounds();

        for (int level = 0;; ++level)
        {
            const auto* items = getItemsForLevel (level);
            if (items == nullptr || items->empty())
                break;

            float maxTextWidth = 0.0f;
            int panelHeight = kBorder * 2;

            for (const auto& item : *items)
            {
                if (! item.separator)
                    maxTextWidth = std::max (maxTextWidth, menuFont.getStringWidthFloat (item.text));
                panelHeight += item.separator ? kSeparatorHeight : kRowHeight;
            }

            const int panelWidth = juce::jmax (kMinWidth,
                                               (int) std::ceil (maxTextWidth) + kHorizPad * 2 + kCheckWidth + kArrowWidth);

            int x = anchor.x;
            int y = anchor.y;

            if (level > 0 && level - 1 < (int) layoutCache.size())
            {
                const auto& prev = layoutCache[(size_t) (level - 1)];
                const int parentIdx = openPath[(size_t) (level - 1)];
                const auto parentRow = (parentIdx >= 0 && parentIdx < (int) prev.itemBounds.size())
                                     ? prev.itemBounds[(size_t) parentIdx]
                                     : prev.bounds;

                x = prev.bounds.getRight() - 1;
                y = parentRow.getY();

                if (x + panelWidth > bounds.getRight())
                    x = juce::jmax (bounds.getX(), prev.bounds.getX() - panelWidth + 1);
            }

            if (x + panelWidth > bounds.getRight())
                x = juce::jmax (bounds.getX(), bounds.getRight() - panelWidth);
            if (y + panelHeight > bounds.getBottom())
                y = juce::jmax (bounds.getY(), bounds.getBottom() - panelHeight);

            PanelLayout panel;
            panel.bounds = juce::Rectangle<int> (x, y, panelWidth, panelHeight);
            panel.itemBounds.reserve (items->size());

            int cy = y + kBorder;
            for (const auto& item : *items)
            {
                const int h = item.separator ? kSeparatorHeight : kRowHeight;
                panel.itemBounds.push_back (juce::Rectangle<int> (x + kBorder, cy, panelWidth - kBorder * 2, h));
                cy += h;
            }

            layoutCache.push_back (std::move (panel));

            if (level >= (int) openPath.size())
                break;

            const int parentIdx = openPath[(size_t) level];
            if (parentIdx < 0 || parentIdx >= (int) items->size() || (*items)[(size_t) parentIdx].children.empty())
                break;
        }
    }

    bool hitTestPanels (juce::Point<int> pos, int& outLevel, int& outIndex) const
    {
        outLevel = -1;
        outIndex = -1;

        for (int level = (int) layoutCache.size() - 1; level >= 0; --level)
        {
            const auto& panel = layoutCache[(size_t) level];
            if (! panel.bounds.contains (pos))
                continue;

            outLevel = level;
            for (int i = 0; i < (int) panel.itemBounds.size(); ++i)
            {
                if (panel.itemBounds[(size_t) i].contains (pos))
                {
                    outIndex = i;
                    return true;
                }
            }
            return true;
        }

        return false;
    }

    void updateHover (juce::Point<int> pos)
    {
        int level = -1, index = -1;
        if (! hitTestPanels (pos, level, index))
            return;

        const auto* items = getItemsForLevel (level);
        if (items == nullptr || index < 0 || index >= (int) items->size())
            return;

        ensureHighlightSize (level + 1);
        highlightIndices[(size_t) level] = index;

        const auto& item = (*items)[(size_t) index];
        if (! item.children.empty() && ! item.disabled && ! item.separator)
        {
            ensureOpenPathSize (level + 1);
            openPath[(size_t) level] = index;

            const auto* childItems = getItemsForLevel (level + 1);
            ensureHighlightSize (level + 2);
            highlightIndices[(size_t) (level + 1)] = childItems ? firstSelectableIndex (*childItems) : -1;
        }
        else
        {
            if ((int) openPath.size() > level)
                openPath.resize ((size_t) level);
            if ((int) highlightIndices.size() > level + 1)
                highlightIndices.resize ((size_t) level + 1);
        }

        rebuildLayout();
        repaint();
    }

    void activateItem (int level, int index)
    {
        const auto* items = getItemsForLevel (level);
        if (items == nullptr || index < 0 || index >= (int) items->size())
            return;

        const auto& item = (*items)[(size_t) index];
        if (item.separator || item.disabled)
            return;

        ensureHighlightSize (level + 1);
        highlightIndices[(size_t) level] = index;

        if (! item.children.empty())
        {
            ensureOpenPathSize (level + 1);
            openPath[(size_t) level] = index;

            const auto* childItems = getItemsForLevel (level + 1);
            ensureHighlightSize (level + 2);
            highlightIndices[(size_t) (level + 1)] = childItems ? firstSelectableIndex (*childItems) : -1;

            rebuildLayout();
            repaint();
            return;
        }

        scheduleFinish (item.resultId, true);
    }

    void scheduleFinish (int result, bool notify)
    {
        if (! menuOpen && ! finishScheduled)
            return;

        scheduledResult = result;
        scheduledNotify = notify;

        if (finishScheduled)
            return;

        finishScheduled = true;
        auto safeThis = juce::Component::SafePointer<GfxMenuOverlay> (this);
        juce::MessageManager::callAsync ([safeThis]
        {
            if (safeThis != nullptr)
                safeThis->flushScheduledFinish();
        });
    }

    void flushScheduledFinish()
    {
        if (! finishScheduled)
            return;

        const int result = scheduledResult;
        const bool notify = scheduledNotify;
        finishScheduled = false;
        scheduledResult = 0;
        scheduledNotify = false;

        finishNow (result, notify);
    }

    void finishNow (int result, bool notify)
    {
        menuOpen = false;
        rootItems.clear();
        openPath.clear();
        highlightIndices.clear();
        layoutCache.clear();
        setVisible (false);

        if (notify && onFinished)
            onFinished (result);
    }

    std::vector<Item> rootItems;
    std::vector<int> openPath;
    std::vector<int> highlightIndices;
    std::vector<PanelLayout> layoutCache;
    juce::Point<int> anchor;
    juce::Font menuFont { juce::Font::getDefaultSansSerifFontName(), 13.0f, juce::Font::plain };
    bool menuOpen = false;
    bool finishScheduled = false;
    int scheduledResult = 0;
    bool scheduledNotify = false;
};


    struct JsfxGfxMenuBridge final : public jsfx_gfx::AsyncMenuPort
    {
        enum class ActiveMode
        {
            none,
            modal,
            nonBlocking
        };

        void setCallbacks (std::function<void()> asyncWakeFn,
                           std::function<void()> quiesceInputFn,
                           std::function<void()> workerWakeFn)
        {
            asyncWake = std::move (asyncWakeFn);
            quiesceInput = std::move (quiesceInputFn);
            workerWake = std::move (workerWakeFn);
        }

        int showMenuModal (const juce::String& description, int x, int y) override
        {
            {
                const std::lock_guard<std::mutex> lock (mutex);
                if (! canStartRequestLocked())
                    return 0;

                activeMode = ActiveMode::modal;
                pendingDescription = description;
                pendingX = x;
                pendingY = y;
                hasPendingOpen = true;
                hasPendingCancel = false;
                hasCompletedResult = false;
                completedResult = 0;
            }

            if (quiesceInput)
                quiesceInput();

            if (asyncWake)
                asyncWake();

            std::unique_lock<std::mutex> lock (mutex);
            completedCv.wait (lock, [this] { return closed || hasCompletedResult; });

            const int result = completedResult;
            clearStateLocked();
            return result;
        }

        int showMenuNonBlockingOpen (const juce::String& description, int x, int y) override
        {
            {
                const std::lock_guard<std::mutex> lock (mutex);
                if (! canStartRequestLocked())
                    return 0;

                activeMode = ActiveMode::nonBlocking;
                pendingDescription = description;
                pendingX = x;
                pendingY = y;
                hasPendingOpen = true;
                hasPendingCancel = false;
                hasCompletedResult = false;
                completedResult = jsfx_gfx::SHOWMENU_NB_NONE_VALUE;
            }

            if (quiesceInput)
                quiesceInput();

            if (asyncWake)
                asyncWake();

            if (workerWake)
                workerWake();

            return 1;
        }

        int showMenuNonBlockingPoll() override
        {
            const std::lock_guard<std::mutex> lock (mutex);
            if (activeMode != ActiveMode::nonBlocking)
                return jsfx_gfx::SHOWMENU_NB_NONE_VALUE;

            if (! hasCompletedResult)
                return jsfx_gfx::SHOWMENU_NB_PENDING_VALUE;

            const int result = completedResult;
            clearStateLocked();
            return result;
        }

        int showMenuNonBlockingCancel() override
        {
            {
                const std::lock_guard<std::mutex> lock (mutex);
                if (activeMode != ActiveMode::nonBlocking || hasCompletedResult || hasPendingCancel)
                    return 0;

                hasPendingCancel = true;
            }

            if (asyncWake)
                asyncWake();

            if (workerWake)
                workerWake();

            return 1;
        }

        bool takePendingOpen (juce::String& description, int& x, int& y)
        {
            const std::lock_guard<std::mutex> lock (mutex);
            if (! hasPendingOpen)
                return false;

            description = pendingDescription;
            x = pendingX;
            y = pendingY;
            hasPendingOpen = false;
            return true;
        }

        bool takePendingCancel()
        {
            const std::lock_guard<std::mutex> lock (mutex);
            if (! hasPendingCancel)
                return false;

            hasPendingCancel = false;
            hasPendingOpen = false;
            pendingDescription.clear();
            pendingX = 0;
            pendingY = 0;
            return true;
        }

        void completeMenu (int rawResult)
        {
            {
                const std::lock_guard<std::mutex> lock (mutex);
                if (activeMode == ActiveMode::none)
                    return;

                completedResult = translateResultForActiveModeLocked (rawResult);
                hasCompletedResult = true;
                hasPendingOpen = false;
                hasPendingCancel = false;
                pendingDescription.clear();
                pendingX = 0;
                pendingY = 0;
            }

            completedCv.notify_all();

            if (workerWake)
                workerWake();

            if (asyncWake)
                asyncWake();
        }

        void close()
        {
            {
                const std::lock_guard<std::mutex> lock (mutex);
                closed = true;
                completedResult = activeMode == ActiveMode::nonBlocking
                                    ? jsfx_gfx::SHOWMENU_NB_CANCELED_VALUE : 0;
                hasCompletedResult = true;
                hasPendingOpen = false;
                hasPendingCancel = false;
                pendingDescription.clear();
            }
            completedCv.notify_all();
        }

        void cancelAll() override
        {
            bool hadActiveMenu = false;
            {
                const std::lock_guard<std::mutex> lock (mutex);
                hadActiveMenu = (activeMode != ActiveMode::none)
                             || hasPendingOpen
                             || hasPendingCancel
                             || hasCompletedResult;

                if (! hadActiveMenu)
                {
                    clearStateLocked();
                }
                else
                {
                    completedResult = (activeMode == ActiveMode::nonBlocking)
                                    ? jsfx_gfx::SHOWMENU_NB_CANCELED_VALUE
                                    : 0;
                    hasCompletedResult = true;
                    hasPendingOpen = false;
                    hasPendingCancel = false;
                    pendingDescription.clear();
                    pendingX = 0;
                    pendingY = 0;
                }
            }

            if (hadActiveMenu)
            {
                completedCv.notify_all();

                if (workerWake)
                    workerWake();

                if (asyncWake)
                    asyncWake();
            }
        }

    private:
        bool canStartRequestLocked() const noexcept
        {
            return ! closed && activeMode == ActiveMode::none
                && ! hasPendingOpen
                && ! hasPendingCancel
                && ! hasCompletedResult;
        }

        int translateResultForActiveModeLocked (int rawResult) const noexcept
        {
            if (activeMode == ActiveMode::nonBlocking)
                return rawResult > 0 ? rawResult : jsfx_gfx::SHOWMENU_NB_CANCELED_VALUE;

            return rawResult > 0 ? rawResult : 0;
        }

        void clearStateLocked()
        {
            activeMode = ActiveMode::none;
            pendingDescription.clear();
            pendingX = 0;
            pendingY = 0;
            completedResult = 0;
            hasCompletedResult = false;
            hasPendingOpen = false;
            hasPendingCancel = false;
        }

        mutable std::mutex mutex;
        std::condition_variable completedCv;
        juce::String pendingDescription;
        int pendingX = 0;
        int pendingY = 0;
        int completedResult = 0;
        bool closed = false; // Never cleared by clearStateLocked/cancelAll.
        bool hasCompletedResult = false;
        bool hasPendingOpen = false;
        bool hasPendingCancel = false;
        ActiveMode activeMode = ActiveMode::none;

        std::function<void()> asyncWake;
        std::function<void()> quiesceInput;
        std::function<void()> workerWake;
    };

