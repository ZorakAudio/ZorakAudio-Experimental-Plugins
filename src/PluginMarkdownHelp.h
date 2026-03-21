#pragma once

#include <juce_gui_extra/juce_gui_extra.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>
#include <vector>

#if defined(__has_include)
 #if __has_include("PluginReadme.h")
  #include "PluginReadme.h"
 #else
  static const char kPluginReadmeMarkdownText[] = "";
 #endif
#else
 static const char kPluginReadmeMarkdownText[] = "";
#endif

namespace za::pluginui
{
inline juce::String getEmbeddedPluginReadmeMarkdown()
{
    return juce::String::fromUTF8 (kPluginReadmeMarkdownText)
        .replace ("\r\n", "\n")
        .replace ("\r", "\n")
        .trim();
}

inline juce::String fallbackReadmeMarkdown (const juce::String& pluginName)
{
    juce::String title = pluginName.trim();
    if (title.isEmpty())
        title = "Plugin";

    return "# " + title + "\n\n"
           "No embedded `README.md` was found for this plugin.\n\n"
           "Each leaf plugin folder should ship a `README.md`; the `?` panel now renders that markdown directly.";
}

inline juce::Font pickSansFont (float height)
{
   #if JUCE_WINDOWS
    juce::Font f ("Segoe UI", height, juce::Font::plain);
    if (f.getTypefaceName().isNotEmpty())
        return f;
   #endif

    return juce::Font (height);
}

inline juce::Font pickMonoFont (float height)
{
    auto name = juce::Font::getDefaultMonospacedFontName();
    juce::Font f (name, height, juce::Font::plain);
    if (f.getTypefaceName().isNotEmpty())
        return f;

    return juce::Font (height, juce::Font::plain);
}

class MarkdownDocumentComponent final : public juce::Component
{
public:
    MarkdownDocumentComponent()
    {
        setOpaque (false);
    }

    void setMarkdownText (const juce::String& markdown)
    {
        sourceMarkdown = markdown;
        blocks = parseMarkdownBlocks (sourceMarkdown);
        layoutWidth = -1;
        totalHeight = 0;
        repaint();
    }

    int preferredHeightForWidth (int width)
    {
        ensureLayout (width);
        return totalHeight;
    }

    void resized() override
    {
        ensureLayout (getWidth());
    }

    void paint (juce::Graphics& g) override
    {
        ensureLayout (getWidth());

        for (const auto& item : laidOutItems)
        {
            if (item.type == BlockType::rule)
            {
                g.setColour (juce::Colours::white.withAlpha (0.16f));
                const float y = (float) item.bounds.getCentreY();
                g.drawLine ((float) item.bounds.getX(), y,
                            (float) item.bounds.getRight(), y, 1.2f);
                continue;
            }

            if (item.panelBounds.isEmpty() == false)
            {
                g.setColour (item.panelFill);
                g.fillRoundedRectangle (item.panelBounds.toFloat(), item.panelCornerRadius);

                g.setColour (item.panelStroke);
                g.drawRoundedRectangle (item.panelBounds.toFloat(), item.panelCornerRadius, 1.0f);
            }

            if (item.quoteBarBounds.isEmpty() == false)
            {
                g.setColour (juce::Colour (0xff89b8ff));
                g.fillRoundedRectangle (item.quoteBarBounds.toFloat(), 2.0f);
            }

            if (item.markerBounds.isEmpty() == false && item.marker.isNotEmpty())
            {
                g.setColour (item.markerColour);
                g.setFont (item.markerFont);
                g.drawText (item.marker,
                            item.markerBounds,
                            juce::Justification::topLeft,
                            false);
            }

            item.layout.draw (g, item.textBounds.toFloat());
        }
    }

private:
    enum class BlockType
    {
        heading,
        paragraph,
        quote,
        listItem,
        code,
        rule,
    };

    struct InlineSpan
    {
        juce::String text;
        bool bold = false;
        bool italic = false;
        bool code = false;
    };

    struct Block
    {
        BlockType type = BlockType::paragraph;
        int headingLevel = 0;
        bool orderedList = false;
        bool tableLike = false;
        juce::String marker;
        juce::String text;
        std::vector<InlineSpan> spans;
    };

    struct LayoutItem
    {
        BlockType type = BlockType::paragraph;
        juce::Rectangle<int> bounds;
        juce::Rectangle<int> textBounds;
        juce::Rectangle<int> panelBounds;
        juce::Rectangle<int> quoteBarBounds;
        juce::Rectangle<int> markerBounds;
        juce::String marker;
        juce::Font markerFont;
        juce::Colour markerColour = juce::Colours::white;
        juce::Colour panelFill = juce::Colours::transparentBlack;
        juce::Colour panelStroke = juce::Colours::transparentBlack;
        float panelCornerRadius = 0.0f;
        juce::TextLayout layout;
    };

    static constexpr int kOuterPadding = 22;
    static constexpr int kMaxReadableWidth = 900;
    static constexpr int kBlockGap = 10;
    static constexpr int kParagraphGap = 14;
    static constexpr int kHeadingGap = 10;
    static constexpr int kRuleGap = 18;
    static constexpr int kListMarkerWidth = 30;
    static constexpr int kQuoteBarWidth = 4;
    static constexpr int kQuoteGap = 14;
    static constexpr int kCodePadX = 12;
    static constexpr int kCodePadY = 10;
    static constexpr int kMinUsableWidth = 220;

    static bool isFenceLine (const juce::String& line)
    {
        return line.trimStart().startsWith ("```");
    }

    static bool isRuleLine (const juce::String& line)
    {
        auto t = line.trim();
        if (t.length() < 3)
            return false;

        juce::juce_wchar ch = 0;
        int count = 0;
        for (auto c : t)
        {
            if (c == ' ' || c == '\t')
                continue;

            if (ch == 0)
                ch = c;

            if (c != ch)
                return false;

            if (c != '-' && c != '*' && c != '_')
                return false;

            ++count;
        }

        return count >= 3;
    }

    static bool looksLikeTableLine (const juce::String& line)
    {
        auto t = line.trim();
        return t.startsWithChar ('|') && t.containsChar ('|');
    }

    static bool isTableSeparatorLine (const juce::String& line)
    {
        if (! looksLikeTableLine (line))
            return false;

        auto row = splitTableRow (line);
        if (row.isEmpty())
            return false;

        for (auto cell : row)
        {
            cell = cell.removeCharacters (":- ");
            if (cell.isNotEmpty())
                return false;
        }

        return true;
    }

    static bool isHeadingLine (const juce::String& line, int& level, juce::String& text)
    {
        auto t = line.trim();
        level = 0;

        while (level < t.length() && t[(size_t) level] == '#')
            ++level;

        if (level <= 0 || level > 6)
            return false;

        if (level >= t.length() || ! juce::CharacterFunctions::isWhitespace (t[(size_t) level]))
            return false;

        text = t.substring (level).trim();
        return text.isNotEmpty();
    }

    static bool parseListLine (const juce::String& line, bool& ordered, juce::String& marker, juce::String& body)
    {
        auto t = line.trimStart();
        ordered = false;
        marker.clear();
        body.clear();

        if (t.startsWith ("* ") || t.startsWith ("- ") || t.startsWith (juce::String::fromUTF8 ("• ")))
        {
            marker = juce::String::fromUTF8 ("•");
            body = t.fromFirstOccurrenceOf (" ", false, false).trimStart();
            return body.isNotEmpty();
        }

        int i = 0;
        while (i < t.length() && juce::CharacterFunctions::isDigit (t[(size_t) i]))
            ++i;

        if (i > 0 && i + 1 < t.length() && t[(size_t) i] == '.' && juce::CharacterFunctions::isWhitespace (t[(size_t) i + 1]))
        {
            ordered = true;
            marker = t.substring (0, i + 1);
            body = t.substring (i + 1).trimStart();
            return body.isNotEmpty();
        }

        return false;
    }

    static juce::StringArray splitTableRow (juce::String row)
    {
        row = row.trim();
        if (row.startsWithChar ('|')) row = row.substring (1);
        if (row.endsWithChar ('|'))   row = row.dropLastCharacters (1);

        juce::StringArray cells;
        cells.addTokens (row, "|", "");
        cells.trim();
        return cells;
    }

    static juce::String padRight (juce::String text, int width)
    {
        while (text.length() < width)
            text << " ";
        return text;
    }

    static juce::String convertTableToMonospace (const juce::StringArray& rawLines)
    {
        std::vector<juce::StringArray> rows;
        rows.reserve ((size_t) rawLines.size());

        for (int i = 0; i < rawLines.size(); ++i)
        {
            if (i == 1 && isTableSeparatorLine (rawLines[i]))
                continue;

            auto cells = splitTableRow (rawLines[i]);
            if (! cells.isEmpty())
                rows.push_back (std::move (cells));
        }

        if (rows.empty())
            return {};

        int numCols = 0;
        for (const auto& row : rows)
            numCols = std::max (numCols, row.size());

        std::vector<int> widths ((size_t) numCols, 0);
        for (const auto& row : rows)
        {
            for (int c = 0; c < row.size(); ++c)
                widths[(size_t) c] = std::max (widths[(size_t) c], row[c].length());
        }

        juce::String out;
        for (size_t r = 0; r < rows.size(); ++r)
        {
            const auto& row = rows[r];
            for (int c = 0; c < numCols; ++c)
            {
                const auto cell = c < row.size() ? row[c].trim() : juce::String();
                out << padRight (cell, widths[(size_t) c]);
                if (c + 1 < numCols)
                    out << " | ";
            }
            out << "\n";

            if (r == 0)
            {
                for (int c = 0; c < numCols; ++c)
                {
                    out << juce::String::repeatedString ("-", widths[(size_t) c]);
                    if (c + 1 < numCols)
                        out << "-+-";
                }
                out << "\n";
            }
        }

        return out.trimEnd();
    }

    static std::vector<InlineSpan> parseInlineSpans (const juce::String& text)
    {
        std::vector<InlineSpan> spans;

        juce::String buffer;
        bool bold = false;
        bool italic = false;
        bool code = false;

        auto flush = [&]()
        {
            if (buffer.isEmpty())
                return;

            spans.push_back (InlineSpan { buffer, bold, italic, code });
            buffer.clear();
        };

        for (int i = 0; i < text.length();)
        {
            const auto c = text[(size_t) i];
            const auto next = (i + 1 < text.length() ? text[(size_t) i + 1] : 0);

            if (! code && c == '*' && next == '*')
            {
                flush();
                bold = ! bold;
                i += 2;
                continue;
            }

            if (c == '`')
            {
                flush();
                code = ! code;
                ++i;
                continue;
            }

            if (! code && (c == '*' || c == '_'))
            {
                const auto prev = (i > 0 ? text[(size_t) i - 1] : 0);
                const bool prevWord = juce::CharacterFunctions::isLetterOrDigit (prev);
                const bool nextWord = juce::CharacterFunctions::isLetterOrDigit (next);

                if (prevWord != nextWord)
                {
                    flush();
                    italic = ! italic;
                    ++i;
                    continue;
                }
            }

            buffer << c;
            ++i;
        }

        flush();

        if (spans.empty())
            spans.push_back (InlineSpan { text, false, false, false });

        return spans;
    }

    static juce::String blockBoundaryProbe (const juce::String& line)
    {
        return line.trim();
    }

    static std::vector<Block> parseMarkdownBlocks (const juce::String& markdown)
    {
        juce::StringArray lines;
        lines.addLines (markdown.replace ("\r\n", "\n").replace ("\r", "\n"));

        std::vector<Block> out;

        for (int i = 0; i < lines.size();)
        {
            const auto rawLine = lines[i];
            const auto trimmed = rawLine.trim();

            if (trimmed.isEmpty())
            {
                ++i;
                continue;
            }

            if (isFenceLine (rawLine))
            {
                juce::String codeText;
                ++i;
                while (i < lines.size() && ! isFenceLine (lines[i]))
                {
                    codeText << lines[i] << "\n";
                    ++i;
                }
                if (i < lines.size() && isFenceLine (lines[i]))
                    ++i;

                Block b;
                b.type = BlockType::code;
                b.text = codeText.trimEnd();
                out.push_back (std::move (b));
                continue;
            }

            if (looksLikeTableLine (trimmed) && i + 1 < lines.size() && isTableSeparatorLine (lines[i + 1]))
            {
                juce::StringArray tableLines;
                tableLines.add (lines[i].trim());
                tableLines.add (lines[i + 1].trim());
                i += 2;
                while (i < lines.size() && looksLikeTableLine (lines[i].trim()))
                {
                    tableLines.add (lines[i].trim());
                    ++i;
                }

                Block b;
                b.type = BlockType::code;
                b.tableLike = true;
                b.text = convertTableToMonospace (tableLines);
                out.push_back (std::move (b));
                continue;
            }

            if (isRuleLine (trimmed))
            {
                Block b;
                b.type = BlockType::rule;
                out.push_back (std::move (b));
                ++i;
                continue;
            }

            int headingLevel = 0;
            juce::String headingText;
            if (isHeadingLine (rawLine, headingLevel, headingText))
            {
                Block b;
                b.type = BlockType::heading;
                b.headingLevel = headingLevel;
                b.text = headingText;
                b.spans = parseInlineSpans (headingText);
                out.push_back (std::move (b));
                ++i;
                continue;
            }

            if (trimmed.startsWithChar ('>'))
            {
                juce::String quoteText;
                while (i < lines.size())
                {
                    auto q = lines[i].trim();
                    if (! q.startsWithChar ('>'))
                        break;

                    q = q.substring (1).trimStart();
                    if (q.isEmpty())
                        quoteText << "\n";
                    else
                        quoteText << q << "\n";
                    ++i;
                }

                Block b;
                b.type = BlockType::quote;
                b.text = quoteText.trimEnd();
                b.spans = parseInlineSpans (b.text.replace ("\n", " "));
                out.push_back (std::move (b));
                continue;
            }

            bool ordered = false;
            juce::String marker;
            juce::String body;
            if (parseListLine (rawLine, ordered, marker, body))
            {
                Block b;
                b.type = BlockType::listItem;
                b.orderedList = ordered;
                b.marker = marker;
                b.text = body;
                b.spans = parseInlineSpans (body);
                out.push_back (std::move (b));
                ++i;
                continue;
            }

            juce::String paragraph;
            while (i < lines.size())
            {
                const auto candidate = lines[i];
                const auto probe = blockBoundaryProbe (candidate);
                if (probe.isEmpty())
                    break;

                int nextHeadingLevel = 0;
                juce::String nextHeadingText;
                bool nextOrdered = false;
                juce::String nextMarker, nextBody;

                if (isFenceLine (candidate)
                    || isRuleLine (probe)
                    || (looksLikeTableLine (probe) && i + 1 < lines.size() && isTableSeparatorLine (lines[i + 1]))
                    || isHeadingLine (candidate, nextHeadingLevel, nextHeadingText)
                    || probe.startsWithChar ('>')
                    || parseListLine (candidate, nextOrdered, nextMarker, nextBody))
                    break;

                if (! paragraph.isEmpty())
                    paragraph << " ";
                paragraph << probe;
                ++i;
            }

            if (paragraph.isNotEmpty())
            {
                Block b;
                b.type = BlockType::paragraph;
                b.text = paragraph;
                b.spans = parseInlineSpans (paragraph);
                out.push_back (std::move (b));
                continue;
            }

            ++i;
        }

        return out;
    }

    static float fontSizeForHeading (int level)
    {
        switch (juce::jlimit (1, 6, level))
        {
            case 1: return 28.0f;
            case 2: return 22.0f;
            case 3: return 18.5f;
            case 4: return 17.0f;
            case 5: return 16.0f;
            default: return 15.5f;
        }
    }

    static juce::Colour bodyColour() noexcept     { return juce::Colour (0xffedf3f9); }
    static juce::Colour headingColour() noexcept  { return juce::Colour (0xfffbfdff); }
    static juce::Colour quoteColour() noexcept    { return juce::Colour (0xffd6e2ee); }
    static juce::Colour codeColour() noexcept     { return juce::Colour (0xffe3f0ff); }
    static juce::Colour subtleColour() noexcept   { return juce::Colour (0xffb7c4d1); }

    static juce::AttributedString makeAttributedString (const std::vector<InlineSpan>& spans,
                                                        float fontSize,
                                                        juce::Colour colour,
                                                        bool baseBold,
                                                        bool baseItalic,
                                                        bool wrapByChar,
                                                        float lineSpacing)
    {
        juce::AttributedString attr;
        attr.setJustification (juce::Justification::topLeft);
        attr.setWordWrap (wrapByChar ? juce::AttributedString::byChar : juce::AttributedString::byWord);
        attr.setLineSpacing (lineSpacing);

        auto baseFont = pickSansFont (fontSize);
        if (baseBold)
            baseFont = baseFont.boldened();
        if (baseItalic)
            baseFont = baseFont.italicised();

        for (const auto& span : spans)
        {
            auto font = baseFont;
            auto spanColour = colour;

            if (span.code)
            {
                font = pickMonoFont (fontSize * 0.95f);
                if (baseBold)
                    font = font.boldened();
                spanColour = codeColour();
            }
            else
            {
                if (span.bold)
                    font = font.boldened();
                if (span.italic)
                    font = font.italicised();
            }

            attr.append (span.text, font, spanColour);
        }

        if (attr.getText().isEmpty())
            attr.append (" ", baseFont, colour);

        return attr;
    }

    static juce::TextLayout buildLayout (const juce::AttributedString& attr, float width)
    {
        juce::TextLayout layout;
        layout.createLayout (attr, width);
        return layout;
    }

    void ensureLayout (int width)
    {
        const int safeWidth = juce::jmax (1, width);
        if (safeWidth == layoutWidth)
            return;

        layoutWidth = safeWidth;
        laidOutItems.clear();

        const int usableW = juce::jmax (kMinUsableWidth, safeWidth - kOuterPadding * 2);
        const int contentW = juce::jmin (usableW, kMaxReadableWidth);
        const int contentX = kOuterPadding + juce::jmax (0, (usableW - contentW) / 2);

        int y = kOuterPadding;

        auto pushLayoutItem = [&] (LayoutItem item, int gapAfter)
        {
            y += item.bounds.getHeight();
            laidOutItems.push_back (std::move (item));
            y += gapAfter;
        };

        for (const auto& block : blocks)
        {
            switch (block.type)
            {
                case BlockType::rule:
                {
                    LayoutItem item;
                    item.type = block.type;
                    item.bounds = { contentX, y + 2, contentW, 12 };
                    pushLayoutItem (std::move (item), kRuleGap);
                    break;
                }

                case BlockType::heading:
                {
                    const auto size = fontSizeForHeading (block.headingLevel);
                    auto attr = makeAttributedString (block.spans, size, headingColour(), true, false, false, 0.10f);
                    auto layout = buildLayout (attr, (float) contentW);

                    LayoutItem item;
                    item.type = block.type;
                    item.textBounds = { contentX, y, contentW, (int) std::ceil (layout.getHeight()) };
                    item.bounds = item.textBounds;
                    item.layout = std::move (layout);
                    pushLayoutItem (std::move (item), kHeadingGap);
                    break;
                }

                case BlockType::paragraph:
                {
                    auto attr = makeAttributedString (block.spans, 15.5f, bodyColour(), false, false, false, 0.20f);
                    auto layout = buildLayout (attr, (float) contentW);

                    LayoutItem item;
                    item.type = block.type;
                    item.textBounds = { contentX, y, contentW, (int) std::ceil (layout.getHeight()) };
                    item.bounds = item.textBounds;
                    item.layout = std::move (layout);
                    pushLayoutItem (std::move (item), kParagraphGap);
                    break;
                }

                case BlockType::quote:
                {
                    const int bodyX = contentX + kQuoteBarWidth + kQuoteGap;
                    const int bodyW = juce::jmax (80, contentW - (bodyX - contentX));
                    auto attr = makeAttributedString (block.spans, 15.0f, quoteColour(), false, true, false, 0.18f);
                    auto layout = buildLayout (attr, (float) bodyW);
                    const int textH = (int) std::ceil (layout.getHeight());

                    LayoutItem item;
                    item.type = block.type;
                    item.textBounds = { bodyX, y + 2, bodyW, textH };
                    item.quoteBarBounds = { contentX + 2, y + 4, kQuoteBarWidth, juce::jmax (18, textH - 4) };
                    item.bounds = { contentX, y, contentW, juce::jmax (textH, item.quoteBarBounds.getHeight()) + 4 };
                    item.layout = std::move (layout);
                    pushLayoutItem (std::move (item), kParagraphGap);
                    break;
                }

                case BlockType::listItem:
                {
                    const int bodyX = contentX + kListMarkerWidth;
                    const int bodyW = juce::jmax (80, contentW - kListMarkerWidth);
                    auto attr = makeAttributedString (block.spans, 15.4f, bodyColour(), false, false, false, 0.18f);
                    auto layout = buildLayout (attr, (float) bodyW);
                    const int textH = (int) std::ceil (layout.getHeight());

                    LayoutItem item;
                    item.type = block.type;
                    item.marker = block.marker;
                    item.markerFont = pickSansFont (15.5f).boldened();
                    item.markerColour = juce::Colour (0xff9dd0ff);
                    item.markerBounds = { contentX, y, kListMarkerWidth - 6, 22 };
                    item.textBounds = { bodyX, y, bodyW, textH };
                    item.bounds = { contentX, y, contentW, juce::jmax (textH, 22) };
                    item.layout = std::move (layout);
                    pushLayoutItem (std::move (item), 8);
                    break;
                }

                case BlockType::code:
                {
                    const int textW = juce::jmax (80, contentW - kCodePadX * 2);
                    auto monoSpans = std::vector<InlineSpan> { InlineSpan { block.text, false, false, true } };
                    auto attr = makeAttributedString (monoSpans, block.tableLike ? 13.3f : 13.6f,
                                                      codeColour(), false, false, true, 0.12f);
                    auto layout = buildLayout (attr, (float) textW);
                    const int textH = (int) std::ceil (layout.getHeight());

                    LayoutItem item;
                    item.type = block.type;
                    item.panelFill = block.tableLike ? juce::Colour (0x18253a52) : juce::Colour (0x1628333f);
                    item.panelStroke = block.tableLike ? juce::Colour (0x50658db3) : juce::Colour (0x3fffffff);
                    item.panelCornerRadius = 8.0f;
                    item.textBounds = { contentX + kCodePadX, y + kCodePadY, textW, textH };
                    item.panelBounds = { contentX, y, contentW, textH + kCodePadY * 2 };
                    item.bounds = item.panelBounds;
                    item.layout = std::move (layout);
                    pushLayoutItem (std::move (item), kParagraphGap);
                    break;
                }
            }
        }

        totalHeight = juce::jmax (y + kOuterPadding, 1);
        setSize (safeWidth, totalHeight);
    }

    juce::String sourceMarkdown;
    std::vector<Block> blocks;
    std::vector<LayoutItem> laidOutItems;
    int layoutWidth = -1;
    int totalHeight = 0;
};

class MarkdownHelpOverlay final : public juce::Component
{
public:
    MarkdownHelpOverlay()
    {
        title.setText ("README", juce::dontSendNotification);
        title.setJustificationType (juce::Justification::centredLeft);
        title.setFont (pickSansFont (17.0f).boldened());
        addAndMakeVisible (title);

        subtitle.setText ("Embedded plugin guide", juce::dontSendNotification);
        subtitle.setJustificationType (juce::Justification::centredLeft);
        subtitle.setFont (pickSansFont (12.5f));
        subtitle.setColour (juce::Label::textColourId, juce::Colour (0xffb7c4d1));
        addAndMakeVisible (subtitle);

        close.setButtonText ("Close");
        close.onClick = [this] { setVisible (false); };
        close.setTooltip ("Hide the README panel");
        addAndMakeVisible (close);

        viewport.setViewedComponent (&document, false);
        viewport.setScrollBarsShown (true, false, true, false);
        viewport.setSingleStepSizes (0, 32);
        addAndMakeVisible (viewport);

        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);
        setInterceptsMouseClicks (true, true);
    }

    void setHeaderTitle (const juce::String& text)
    {
        const auto trimmed = text.trim();
        title.setText (trimmed.isNotEmpty() ? trimmed : juce::String ("README"), juce::dontSendNotification);
    }

    void setMarkdownText (const juce::String& markdown)
    {
        document.setMarkdownText (markdown);
        updateDocumentBounds();
        viewport.setViewPosition (0, 0);
    }

    bool keyPressed (const juce::KeyPress& key) override
    {
        if (key == juce::KeyPress::escapeKey)
        {
            setVisible (false);
            return true;
        }

        int x = viewport.getViewPositionX();
        int y = viewport.getViewPositionY();
        const int step = 36;
        const int page = juce::jmax (step, viewport.getMaximumVisibleHeight() - 48);
        const int maxY = juce::jmax (0, document.getHeight() - viewport.getMaximumVisibleHeight());

        if (key == juce::KeyPress::upKey)         { viewport.setViewPosition (x, juce::jmax (0, y - step)); return true; }
        if (key == juce::KeyPress::downKey)       { viewport.setViewPosition (x, juce::jmin (maxY, y + step)); return true; }
        if (key == juce::KeyPress::pageUpKey)     { viewport.setViewPosition (x, juce::jmax (0, y - page)); return true; }
        if (key == juce::KeyPress::pageDownKey)   { viewport.setViewPosition (x, juce::jmin (maxY, y + page)); return true; }
        if (key == juce::KeyPress::homeKey)       { viewport.setViewPosition (x, 0); return true; }
        if (key == juce::KeyPress::endKey)        { viewport.setViewPosition (x, maxY); return true; }

        return false;
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::black.withAlpha (0.60f));

        const auto panel = panelBounds.toFloat();
        g.setColour (juce::Colour (0xff20272f).withAlpha (0.98f));
        g.fillRoundedRectangle (panel, 12.0f);

        g.setColour (juce::Colours::white.withAlpha (0.18f));
        g.drawRoundedRectangle (panel, 12.0f, 1.0f);

    }

    void resized() override
    {
        auto r = getLocalBounds();
        const int marginX = juce::jlimit (18, 40, getWidth() / 20);
        const int marginY = juce::jlimit (18, 40, getHeight() / 20);
        auto panel = r.reduced (marginX, marginY);

        panelBounds = panel;

        auto inner = panel.reduced (18, 16);
        auto header = inner.removeFromTop (34);
        auto closeArea = header.removeFromRight (92);
        close.setBounds (closeArea.withTrimmedTop (2));

        auto titles = header;
        title.setBounds (titles.removeFromTop (19));
        subtitle.setBounds (titles);

        inner.removeFromTop (10);
        viewport.setBounds (inner);
        updateDocumentBounds();
    }

    void mouseUp (const juce::MouseEvent& e) override
    {
        if (! panelBounds.contains (e.getPosition()))
            setVisible (false);
    }

private:
    void updateDocumentBounds()
    {
        if (viewport.getWidth() <= 0 || viewport.getHeight() <= 0)
            return;

        const int docW = juce::jmax (1, viewport.getMaximumVisibleWidth());
        const int docH = document.preferredHeightForWidth (docW);
        document.setBounds (0, 0, docW, docH);
    }

    juce::Rectangle<int> panelBounds;
    juce::Label title;
    juce::Label subtitle;
    juce::TextButton close;
    juce::Viewport viewport;
    MarkdownDocumentComponent document;
};

inline juce::String firstMarkdownHeading (const juce::String& markdown)
{
    juce::StringArray lines;
    lines.addLines (markdown.replace ("\r\n", "\n").replace ("\r", "\n"));

    for (auto line : lines)
    {
        line = line.trim();
        if (! line.startsWithChar ('#'))
            continue;

        int count = 0;
        while (count < line.length() && line[(size_t) count] == '#')
            ++count;

        if (count > 0 && count < line.length())
        {
            auto heading = line.substring (count).trim();
            if (heading.isNotEmpty())
                return heading;
        }
    }

    return {};
}

struct ScaledSectionLayout
{
    int initialWidth = 0;
    int initialHeight = 0;
    int minWidth = 0;
    int minHeight = 0;
    int initialGfxHeight = 0;
    int initialControlsHeight = 0;
    int minimumControlsHeight = 0;
};

inline ScaledSectionLayout planEditorSections (int topBarHeight,
                                               int minEditorWidth,
                                               int maxEditorWidth,
                                               int maxEditorHeight,
                                               bool hasControls,
                                               int controlsPrefWidth,
                                               int controlsPrefHeight,
                                               bool hasGfx,
                                               int gfxPreferredWidth,
                                               int gfxPreferredHeight,
                                               int sectionGap,
                                               int minimumGfxHeight = 240)
{
    ScaledSectionLayout plan {};

    const int controlsW = hasControls ? controlsPrefWidth : 0;
    const int controlsH = hasControls ? controlsPrefHeight : 0;
    const int gfxW = hasGfx ? juce::jmax (1, gfxPreferredWidth) : 0;
    const int gfxH = hasGfx ? juce::jmax (1, gfxPreferredHeight) : 0;

    plan.minimumControlsHeight = hasControls ? juce::jmin (controlsH, 180) : 0;

    plan.initialWidth = juce::jlimit (minEditorWidth,
                                      maxEditorWidth,
                                      juce::jmax (minEditorWidth, juce::jmax (controlsW, gfxW)));

    if (hasGfx && gfxW > 0)
    {
        const double scale = (double) plan.initialWidth / (double) gfxW;
        plan.initialGfxHeight = juce::jmax (minimumGfxHeight,
                                            (int) std::llround ((double) gfxH * scale));
    }
    else
    {
        plan.initialGfxHeight = 0;
    }

    if (hasControls)
        plan.initialControlsHeight = controlsH;

    const int gap = (hasControls && hasGfx) ? sectionGap : 0;
    int total = topBarHeight + plan.initialControlsHeight + gap + plan.initialGfxHeight;

    if (hasGfx && total > maxEditorHeight)
    {
        const int availableForGfx = juce::jmax (minimumGfxHeight,
                                                maxEditorHeight - topBarHeight - gap - plan.initialControlsHeight);
        plan.initialGfxHeight = juce::jmin (plan.initialGfxHeight, availableForGfx);
        total = topBarHeight + plan.initialControlsHeight + gap + plan.initialGfxHeight;
    }

    if (hasControls && total > maxEditorHeight)
    {
        const int remainingForControls = juce::jmax (plan.minimumControlsHeight,
                                                     maxEditorHeight - topBarHeight - gap - plan.initialGfxHeight);
        plan.initialControlsHeight = juce::jmin (controlsH, remainingForControls);
        total = topBarHeight + plan.initialControlsHeight + gap + plan.initialGfxHeight;
    }

    plan.minWidth = juce::jlimit (minEditorWidth,
                                  maxEditorWidth,
                                  juce::jmax (minEditorWidth,
                                               juce::jmax (hasControls ? juce::jmin (controlsW, 720) : 0,
                                                           hasGfx ? juce::jmin (gfxW, 640) : 0)));

    const int minGfx = hasGfx ? juce::jmin (plan.initialGfxHeight, juce::jmax (minimumGfxHeight, gfxH / 2)) : 0;
    plan.minHeight = topBarHeight + (hasControls ? plan.minimumControlsHeight : 0) + gap + minGfx;

    plan.initialHeight = juce::jlimit (plan.minHeight,
                                       juce::jmax (plan.minHeight, maxEditorHeight),
                                       total);

    return plan;
}

} // namespace za::pluginui
