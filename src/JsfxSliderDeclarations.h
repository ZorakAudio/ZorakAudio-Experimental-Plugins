#pragma once
#include <juce_core/juce_core.h>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
static juce::String sanitizeId (const juce::String& s)
{
    juce::String out;
    for (auto c : s)
    {
        if (juce::CharacterFunctions::isLetterOrDigit (c))
            out << c;
        else if (c == ' ' || c == '-' || c == '_')
            out << '_';
    }

    if (out.isEmpty())
        out = "param";

    return out;
}

struct JsfxSliderDecl
{
    int index0 = 0;
    juce::String id;
    juce::String name;
    juce::String varName; // optional slider alias variable (e.g. slider1:thresh_db=...)
    float def  = 0.0f;
    // Numeric domain is always stored ascending for JUCE/clamping, while
    // rangeStart/rangeEnd preserve the JSFX declaration order. REAPER uses
    // that order to determine slider direction, so <0,-340,...> is a
    // right-to-left numeric range rather than an ascending [-340,0] control.
    float min  = 0.0f;
    float max  = 1.0f;
    float rangeStart = 0.0f;
    float rangeEnd   = 1.0f;
    float step = 0.001f;
    bool reversed = false;
    float shapeModifier = 0.0f; // for :log=mid or :sqr=exp

    enum class Shape { Linear = 0, Log = 1, Sqr = 2 };
    Shape shape = Shape::Linear;

    // Optional enum choices parsed from "step{A,B,C}" syntax
    juce::StringArray choices;
    bool isChoice = false;

    // DSP-JSFX extension: string input slider.
    // Syntax:
    //   slider1:#bus_name="main"<string>Bus Name
    // The alias variable receives an opaque runtime string handle.
    bool isString = false;
    juce::String stringDefault;

    // Optional UI metadata parsed from JSFX comments
    juce::String tooltip;

    bool hidden = false;
};

struct JsfxFileDecl
{
    int index0 = 0;
    juce::String name;        // UI label
    juce::String description; // optional tooltip
    juce::String token;       // raw token after comma in filename:N,token
    juce::String defaultPath; // heuristic default path (may be empty)

    // DSP-JSFX extension. A plain REAPER `filename:N,...` declaration remains
    // a native JSFX resource declaration. Only an immediately preceding
    // `// #FILE: ...` opts the slot into the enhanced ZA file-import UI and
    // editor-level drag/drop handling.
    bool enhancedImport = false;
};

static inline std::string trimAscii (std::string s)
{
    while (! s.empty() && std::isspace ((unsigned char) s.front()))
        s.erase (s.begin());
    while (! s.empty() && std::isspace ((unsigned char) s.back()))
        s.pop_back();
    return s;
}

// Split a JSFX "<min,max,step{...},skew>" range string on commas, but ignore commas inside { }.
// This is required because the enum-list itself is comma-separated.
static std::vector<std::string> splitTopLevelCommas (const std::string& s)
{
    std::vector<std::string> parts;
    std::string cur;
    int braceDepth = 0;

    for (char c : s)
    {
        if (c == '{')
            ++braceDepth;
        else if (c == '}' && braceDepth > 0)
            --braceDepth;

        if (c == ',' && braceDepth == 0)
        {
            parts.push_back (trimAscii (cur));
            cur.clear();
        }
        else
        {
            cur.push_back (c);
        }
    }

    parts.push_back (trimAscii (cur));
    return parts;
}

static bool parseFloat (const std::string& s, float& out)
{
    char* end = nullptr;
    const double v = std::strtod (s.c_str(), &end);
    if (end == s.c_str())
        return false;
    out = (float) v;
    return true;
}

static juce::String parseJsfxStringDefaultToken (std::string tok)
{
    tok = trimAscii (tok);
    if (tok.size() >= 2)
    {
        const char quote = tok.front();
        if ((quote == '"' || quote == '\'') && tok.back() == quote)
        {
            std::string out;
            out.reserve (tok.size() - 2);
            for (size_t i = 1; i + 1 < tok.size(); ++i)
            {
                char c = tok[i];
                if (c == '\\' && i + 1 < tok.size() - 1)
                {
                    const char e = tok[++i];
                    if (e == 'n') out.push_back ('\n');
                    else if (e == 'r') out.push_back ('\r');
                    else if (e == 't') out.push_back ('\t');
                    else out.push_back (e);
                }
                else
                {
                    out.push_back (c);
                }
            }
            return juce::String::fromUTF8 (out.c_str());
        }
    }
    return juce::String::fromUTF8 (tok.c_str());
}

// sliderN:DEF<MIN,MAX,STEP{,SKEW}>Label
// plus UI metadata comments:
//   // #TOOLTIP: ...  (applies to the next slider line)
//   // #HELP: ...     (deprecated; README.md is now used for the ? help panel)
static std::vector<JsfxSliderDecl> parseJsfxSliderDecls (const char* jsfxText, juce::String* outHelpText = nullptr)
{
    std::vector<JsfxSliderDecl> out;
    if (! jsfxText)
        return out;

    const std::regex reSlider (
        R"(^\s*slider\s*([0-9]{1,2})\s*:\s*([^<\r\n;]+)\s*(?:<\s*([^>]+)\s*>)?\s*(.*)$)",
        std::regex::ECMAScript);

    const std::regex reTooltip (R"(^\s*\/\/\s*#TOOLTIP:\s*(.*)$)", std::regex::ECMAScript);
    const std::regex reHelp    (R"(^\s*\/\/\s*#HELP:\s*(.*)$)",    std::regex::ECMAScript);

    juce::String pendingTooltip;
    juce::String helpAccum;

    std::string text (jsfxText);
    size_t start = 0;

    while (start < text.size())
    {
        size_t end = text.find_first_of ("\r\n", start);
        if (end == std::string::npos)
            end = text.size();

        std::string line = text.substr (start, end - start);

        size_t next = end;
        while (next < text.size() && (text[next] == '\r' || text[next] == '\n'))
            ++next;
        start = next;

        // --- UI metadata comments ---
        {
            std::smatch m;
            if (std::regex_match (line, m, reHelp))
            {
                auto part = juce::String::fromUTF8 (m[1].str().c_str()).trimEnd();
                if (! part.isEmpty())
                {
                    if (! helpAccum.isEmpty())
                        helpAccum << "\n";
                    helpAccum << part;
                }
                continue;
            }

            if (std::regex_match (line, m, reTooltip))
            {
                pendingTooltip = juce::String::fromUTF8 (m[1].str().c_str()).trim();
                continue;
            }
        }

        // --- Slider declarations ---
        std::smatch m;
        if (! std::regex_match (line, m, reSlider))
            continue;

        const int sliderN = std::atoi (m[1].str().c_str());
        if (sliderN < 1 || sliderN > 256)
            continue;

        JsfxSliderDecl d;
        d.index0 = sliderN - 1;
        d.id = "slider" + juce::String (sliderN);

        // Slider declarations can optionally bind a variable name:
        //   slider1:thresh_db=-40<-80,0,0.1>Threshold (dB)
        // In JSFX, that variable reflects the slider value.
        std::string defTokFull = trimAscii (m[2].str());
        std::string varTok;
        std::string defTok = defTokFull;

        if (const auto eq = defTokFull.rfind ('='); eq != std::string::npos)
        {
            varTok = trimAscii (defTokFull.substr (0, eq));
            defTok = trimAscii (defTokFull.substr (eq + 1));
        }

        float def = 0.0f;
        if (! parseFloat (defTok, def))
            def = 0.0f;

        d.def = def;
        d.varName = juce::String::fromUTF8 (varTok.c_str()).trim();

        if (m[3].matched)
        {
            const auto rangeKind = juce::String::fromUTF8 (trimAscii (m[3].str()).c_str()).trim().toLowerCase();
            if (rangeKind == "string" || rangeKind == "str" || rangeKind == "text")
            {
                d.isString = true;
                d.isChoice = false;
                d.choices.clear();
                d.stringDefault = parseJsfxStringDefaultToken (defTok);
            }
        }

        if (! d.isString && d.varName.startsWithChar ('#'))
        {
            d.isString = true;
            d.stringDefault = parseJsfxStringDefaultToken (defTok);
        }

        if (! d.isString && m[3].matched)
        {
            std::string r = m[3].str();
            // IMPORTANT: do NOT naively split on ',' here.
            // JSFX enum syntax is step{A,B,C} which contains commas.
            const std::vector<std::string> parts = splitTopLevelCommas (r);

            float vmin = 0.0f, vmax = 1.0f, vstep = 0.001f;
            if (parts.size() >= 2)
            {
                if (! parseFloat (parts[0], vmin)) vmin = 0.0f;
                if (! parseFloat (parts[1], vmax)) vmax = 1.0f;
            }

            if (parts.size() >= 3)
            {
                // STEP token may be like "1{Eco,Moderate,High}".
                // Extract optional {choices} and parse the numeric prefix.
                std::string stepTok = parts[2];

                const auto bracePos = stepTok.find ('{');
                if (bracePos != std::string::npos)
                {
                    const auto closePos = stepTok.find ('}', bracePos + 1);
                    if (closePos != std::string::npos)
                    {
                        const auto inside = stepTok.substr (bracePos + 1, closePos - (bracePos + 1));
                        auto sInside = juce::String::fromUTF8 (inside.c_str());

                        juce::StringArray labels;
                        labels.addTokens (sInside, ",", "");
                        labels.trim();
                        labels.removeEmptyStrings();

                        if (labels.size() > 0)
                        {
                            d.choices = labels;
                            d.isChoice = true;
                        }
                    }

                    // Strip "{...}" for numeric parsing
                    stepTok = stepTok.substr (0, bracePos);
                    stepTok = trimAscii (stepTok);
                }

                // STEP token may contain a curve tag, e.g. "0.001:sqr" or "1:log".
                // Parse the numeric prefix + optional ":log"/":sqr".
                JsfxSliderDecl::Shape shape = JsfxSliderDecl::Shape::Linear;
                float shapeMod = 0.0f;

                std::string curveTok = stepTok;

                // split numeric part vs tag part
                if (const auto colon = curveTok.find (':'); colon != std::string::npos)
                {
                    std::string tag = trimAscii (curveTok.substr (colon + 1));
                    curveTok = trimAscii (curveTok.substr (0, colon));

                    // tag can be "log" or "log=1000" or "sqr=2"
                    std::string tagBase = tag;
                    if (const auto eq = tag.find ('='); eq != std::string::npos)
                    {
                        tagBase = trimAscii (tag.substr (0, eq));

                        float tmp = 0.0f;
                        if (parseFloat (trimAscii (tag.substr (eq + 1)), tmp))
                            shapeMod = tmp;
                    }

                    if (tagBase == "log") shape = JsfxSliderDecl::Shape::Log;
                    else if (tagBase == "sqr") shape = JsfxSliderDecl::Shape::Sqr;
                }

                // numeric step
                if (curveTok.empty())
                    vstep = 1.0f;
                else if (! parseFloat (curveTok, vstep))
                    vstep = 1.0f;

                d.shape = shape;
                d.shapeModifier = shapeMod;
            }

            d.rangeStart = vmin;
            d.rangeEnd   = vmax;
            d.reversed   = vmax < vmin;
            d.min        = juce::jmin (vmin, vmax);
            d.max        = juce::jmax (vmin, vmax);
            d.step       = (vstep > 0.0f ? vstep : 0.001f);
            d.def        = juce::jlimit (d.min, d.max, d.def);
        }

        auto label = juce::String::fromUTF8 (m[4].str().c_str()).trim();
        if (label.isEmpty())
            label = "Slider " + juce::String (sliderN);
        
            // Hide convention: label begins with '-' (optionally "- ").
            if (label.startsWithChar ('-'))
            {
                d.hidden = true;
                label = label.substring (1).trimStart(); // remove '-' and any following whitespace
                if (label.isEmpty())
                    label = "Slider " + juce::String (sliderN);
            }

        d.name = label;
        d.tooltip = pendingTooltip;
        pendingTooltip.clear();

        out.push_back (d);
    }

    if (outHelpText)
        *outHelpText = helpAccum;

    std::sort (out.begin(), out.end(),
               [] (const JsfxSliderDecl& a, const JsfxSliderDecl& b) { return a.index0 < b.index0; });

    out.erase (std::unique (out.begin(), out.end(),
                            [] (const JsfxSliderDecl& a, const JsfxSliderDecl& b) { return a.index0 == b.index0; }),
               out.end());

    return out;
}


    static inline float clamp01f (float x) noexcept
    {
        return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    }

    static inline float sgnf (float x) noexcept { return x >= 0.0f ? 1.0f : -1.0f; }

    // YSFX-style sqr mapping generalized for bipolar/unipolar.
    // We use modifier=2 for plain ":sqr".
    static float curveFrom01_sqr (float t, float lo, float hi, float modifier = 2.0f) noexcept
    {
        t = clamp01f(t);
        if (hi == lo) return lo;
        if (modifier <= 0.0f) modifier = 2.0f;

        const float inv = 1.0f / modifier;

        const float imax = sgnf(hi) * std::pow(std::abs(hi), inv);
        const float imin = sgnf(lo) * std::pow(std::abs(lo), inv);

        const float interp = t * (imax - imin) + imin;
        return sgnf(interp) * std::pow(std::abs(interp), modifier);
    }

    static float curveTo01_sqr (float v, float lo, float hi, float modifier = 2.0f) noexcept
    {
        if (hi == lo) return 0.0f;
        if (modifier <= 0.0f) modifier = 2.0f;

        // Clamp against the numeric domain, independent of declaration order.
        const float domainLo = juce::jmin (lo, hi);
        const float domainHi = juce::jmax (lo, hi);
        v = juce::jlimit (domainLo, domainHi, v);

        const float inv = 1.0f / modifier;

        const float imax = sgnf(hi) * std::pow(std::abs(hi), inv);
        const float imin = sgnf(lo) * std::pow(std::abs(lo), inv);

        const float interp = sgnf(v) * std::pow(std::abs(v), inv);

        const float denom = (imax - imin);
        if (std::abs(denom) < 1.0e-20f) return 0.0f;

        return clamp01f((interp - imin) / denom);
    }

    static float curveFrom01_log (float t, float lo, float hi, float modifier) noexcept
    {
        t = clamp01f (t);
        if (hi == lo) return lo;

        // modifier == 0 -> classic exp interpolation (needs >0 domain)
        if (modifier == 0.0f)
        {
            if (lo <= 0.0001f || hi <= 0.0001f)
                return lo + t * (hi - lo);

            return lo * std::exp ((std::log (hi) - std::log (lo)) * t);
        }

        // modifier != 0 -> midpoint-based curve (works with lo==0)
        const float diff = hi - lo;
        if (std::abs (diff) < 1.0e-20f) return lo;
        if (std::abs (modifier - lo) < 1.0e-20f) return lo + t * diff;

        const float m = (modifier - lo) / diff;
        if (std::abs (m) < 1.0e-20f) return lo + t * diff;

        float mm1 = (m - 1.0f) / m;
        mm1 *= mm1;

        const float denom = (mm1 - 1.0f);
        if (std::abs (denom) < 1.0e-20f) return lo + t * diff;

        const float prefactor = diff / denom;
        return prefactor * (std::pow (std::abs (mm1), t) - 1.0f) + lo;
    }

    static float curveTo01_log (float v, float lo, float hi, float modifier) noexcept
    {
        if (hi == lo) return 0.0f;

        const float domainLo = juce::jmin (lo, hi);
        const float domainHi = juce::jmax (lo, hi);
        v = juce::jlimit (domainLo, domainHi, v);

        if (modifier == 0.0f)
        {
            if (lo <= 0.0001f || hi <= 0.0001f)
                return clamp01f ((v - lo) / (hi - lo));

            const float denom = (std::log (hi) - std::log (lo));
            if (std::abs (denom) < 1.0e-20f) return 0.0f;

            return clamp01f ((std::log (v) - std::log (lo)) / denom);
        }

        const float diff = hi - lo;
        if (std::abs (diff) < 1.0e-20f) return 0.0f;
        if (std::abs (modifier - lo) < 1.0e-20f) return clamp01f ((v - lo) / diff);

        const float m = (modifier - lo) / diff;
        if (std::abs (m) < 1.0e-20f) return clamp01f ((v - lo) / diff);

        float mm1 = (m - 1.0f) / m;
        mm1 *= mm1;

        const float base = std::abs (mm1);
        const float logBase = std::log (base);
        if (base <= 0.0f || std::abs (logBase) < 1.0e-20f)
            return clamp01f ((v - lo) / diff);

        const float inv_prefactor = (mm1 - 1.0f) / diff;
        const float inside = (v - lo) * inv_prefactor + 1.0f;
        if (inside <= 0.0f) return 0.0f;

        return clamp01f (std::log (std::abs (inside)) / logBase);
    }

static std::vector<JsfxFileDecl> parseJsfxFilenameDecls (const char* jsfxText)
{
    std::vector<JsfxFileDecl> out;
    if (! jsfxText)
        return out;

    const std::regex reFilename (
        R"(^\s*filename\s*:\s*([0-9]{1,3})\s*,\s*([^\r\n;]+)\s*$)",
        std::regex::ECMAScript);

    // Optional UI metadata comment (applies to the next filename line):
    //   // #FILE: Name { optional description }
    const std::regex reFileMeta (R"(^\s*\/\/\s*#FILE:\s*(.*)$)", std::regex::ECMAScript);

    juce::String pendingName;
    juce::String pendingDesc;
    bool pendingEnhancedImport = false;

    std::string text (jsfxText);
    size_t start = 0;

    while (start < text.size())
    {
        size_t end = text.find_first_of ("\r\n", start);
        if (end == std::string::npos)
            end = text.size();

        std::string line = text.substr (start, end - start);

        size_t next = end;
        while (next < text.size() && (text[next] == '\r' || text[next] == '\n'))
            ++next;
        start = next;

        // Strip JSFX ';' comments
        if (auto semi = line.find (';'); semi != std::string::npos)
            line = line.substr (0, semi);

        // Metadata comment
        {
            std::smatch m;
            if (std::regex_match (line, m, reFileMeta))
            {
                // The presence of #FILE itself is the opt-in. Keep that fact
                // even when the author leaves the display label empty.
                pendingEnhancedImport = true;

                auto meta = juce::String::fromUTF8 (m[1].str().c_str()).trim();
                if (! meta.isEmpty())
                {
                    juce::String name = meta;
                    juce::String desc;

                    const int bracePos = meta.indexOfChar ('{');
                    if (bracePos >= 0)
                    {
                        name = meta.substring (0, bracePos).trim();

                        const int closePos = meta.lastIndexOfChar ('}');
                        if (closePos > bracePos)
                            desc = meta.substring (bracePos + 1, closePos).trim();
                        else
                            desc = meta.substring (bracePos + 1).trim();
                    }

                    pendingName = name;
                    pendingDesc = desc;
                }
                continue;
            }
        }

        // filename:N,token
        std::smatch m;
        if (! std::regex_match (line, m, reFilename))
            continue;

        const int idx0 = std::atoi (m[1].str().c_str());
        if (idx0 < 0 || idx0 > 63)
            continue;

        JsfxFileDecl d;
        d.index0 = idx0;
        d.enhancedImport = pendingEnhancedImport;

        const auto token = juce::String::fromUTF8 (m[2].str().c_str()).trim();
        d.token = token;

        // Heuristic: treat the token as a default path if it looks like a filename/path.
        const bool looksLikePath = token.containsChar ('.') || token.containsChar ('/') || token.containsChar ('\\');
        d.defaultPath = looksLikePath ? token : juce::String();

        if (! pendingName.isEmpty())
        {
            d.name = pendingName;
            d.description = pendingDesc;
        }
        else
        {
            d.name = token.isNotEmpty() ? token : ("File " + juce::String (idx0));
            d.description = {};
        }

        pendingName.clear();
        pendingDesc.clear();
        pendingEnhancedImport = false;

        out.push_back (d);
    }

    std::sort (out.begin(), out.end(),
               [] (const JsfxFileDecl& a, const JsfxFileDecl& b) { return a.index0 < b.index0; });

    out.erase (std::unique (out.begin(), out.end(),
                            [] (const JsfxFileDecl& a, const JsfxFileDecl& b) { return a.index0 == b.index0; }),
               out.end());

    return out;
}
