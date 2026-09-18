#pragma once
#include "ZAAudioImportRecipe.h"
#include <condition_variable>
#include <mutex>

namespace za::fileimport
{
struct ImportPreviewSource
{
    AudioFileData audio;
    SourceFingerprint fingerprint;
    extractor::Analysis features;
    extractor::WaveformPyramid waveform;
};

// One owned worker, with a replaceable pending task. No detached tasks can
// outlive the dialog/plugin; cancellation is checked during decode/analysis.
class ImportPreviewWorker
{
public:
    ImportPreviewWorker() : thread ([this] { run(); }) {}
    ~ImportPreviewWorker()
    {
        { std::lock_guard<std::mutex> lock (mutex); stopping = true; pending = {}; }
        condition.notify_one();
        if (thread.joinable()) thread.join();
    }
    void submit (std::function<void()> task)
    {
        { std::lock_guard<std::mutex> lock (mutex); pending = std::move (task); }
        condition.notify_one();
    }
private:
    void run()
    {
        for (;;)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock (mutex);
                condition.wait (lock, [this] { return stopping || (bool) pending; });
                if (stopping) return;
                task = std::move (pending); pending = {};
            }
            // Tasks report ordinary errors themselves. Never unwind out of an
            // owned thread if a callback/allocation fails before that handler.
            try { task(); } catch (...) {}
        }
    }
    std::mutex mutex;
    std::condition_variable condition;
    std::function<void()> pending;
    bool stopping = false;
    std::thread thread;
};

class WaveformPreview final : public juce::Component
{
public:
    using SegmentSelectCallback = std::function<void (int)>;
    using SegmentBoundaryCallback = std::function<void (int, bool, int)>;
    using SegmentDragFinishedCallback = std::function<void()>;
    using SegmentCreatedCallback = std::function<void (int, int)>;

    void setCallbacks (SegmentSelectCallback selectCb,
                       SegmentBoundaryCallback boundaryCb,
                       SegmentDragFinishedCallback dragFinishedCb = {},
                       SegmentCreatedCallback createdCb = {})
    {
        onSegmentSelected = std::move (selectCb);
        onSegmentBoundaryMoved = std::move (boundaryCb);
        onSegmentDragFinished = std::move (dragFinishedCb);
        onSegmentCreated = std::move (createdCb);
    }

    void setSelectedSegment (int index)
    {
        selectedSegment = index;
        repaint();
    }

    void setSource (std::shared_ptr<const ImportPreviewSource> sourceIn,
                    std::shared_ptr<const ImportPreviewSource> processedIn,
                    std::vector<SegmentRegion> segmentsIn, bool segmentationIn,
                    juce::String statusIn, std::vector<extractor::Region> reviewIn = {})
    {
        const bool newSource = sourceData != sourceIn;
        sourceData = std::move (sourceIn); processedData = std::move (processedIn);
        sampleRate = sourceData != nullptr ? sourceData->audio.sampleRate : 0.0;
        segmentationPreview = segmentationIn; status = std::move (statusIn);
        reviewSpans = std::move (reviewIn);
        setSegments (std::move (segmentsIn));
        if (newSource) { cursorSample = -1; resetZoomToFull(); } else clampZoomRange();
        repaint();
    }

    void setSegments (std::vector<SegmentRegion> segmentsIn)
    {
        segments = std::move (segmentsIn);
        keptSpans.clear(); keptSamples = 0;
        for (const auto& s : segments)
            if (s.enabled && s.length() > 0 && s.length() <= std::numeric_limits<int>::max() - keptSamples)
            {
                keptSpans.push_back ({ keptSamples, keptSamples + s.length(), s.startSample });
                keptSamples += s.length();
            }
        repaint();
    }
    void setSnapping (bool enabled) { snapping = enabled; }
    int getCursorSample() const { return cursorSample; }
    void clearReviewFocus() { focusStart = focusEnd = -1; repaint(); }
    void setReviewFocus (int start, int end) { focusStart = start; focusEnd = end; reveal (start, end); repaint(); }
    void reveal (int start, int end)
    {
        if (start >= getVisibleStartSample() && end <= getVisibleEndSample()) return;
        const int span = juce::jmax (end - start + 2 * (int) (sampleRate * 0.15), getVisibleEndSample() - getVisibleStartSample());
        visibleStartSample = juce::jmax (0, start - (span - (end - start)) / 2);
        visibleEndSample = visibleStartSample + span; clampZoomRange(); repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff15191d));

        auto r = getLocalBounds().reduced (8);
        auto kept = getKeptPanelBoundsFor (r);
        auto source = r;
        source.removeFromBottom (kept.getHeight());
        source.removeFromBottom (8);

        drawWave (g, source, originalBuffer(), segmentationPreview ? "Source / Editable Cuts" : "Before", true);
        drawWave (g, kept, processedBuffer(), segmentationPreview ? "Kept regions (fades / gain applied at playback and import)" : "After", false);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        draggingSegment = -1;
        draggingStart = false;
        creatingSegment = false;
        scrollingZoom = false;

        if (! segmentationPreview || originalBuffer().getNumSamples() <= 0)
            return;

        const auto wave = getSourceWaveBounds();
        if (! wave.contains (e.getPosition()))
            return;

        cursorSample = sampleFromX (e.position.x);
        focusStart = focusEnd = -1;
        if (e.mods.isMiddleButtonDown())
        {
            const int start = getVisibleStartSample();
            const int end = getVisibleEndSample();
            if (end > start && end - start < originalBuffer().getNumSamples())
            {
                scrollingZoom = true;
                scrollDragStartX = e.position.x;
                scrollDragStartSample = start;
                scrollDragVisibleSpan = end - start;
            }
            return;
        }

        if (e.mods.isCtrlDown() || e.mods.isCommandDown())
        {
            creatingSegment = true;
            createStartSample = sampleFromX (e.position.x);
            createEndSample = createStartSample;
            repaint();
            return;
        }

        const int hit = findSegmentAtX (e.position.x);
        if (hit >= 0)
        {
            selectedSegment = hit;
            if (onSegmentSelected)
                onSegmentSelected (hit);

            const int x = (int) std::lround (e.position.x);
            const int startX = xFromSample (segments[(size_t) hit].startSample);
            const int endX = xFromSample (segments[(size_t) hit].endSample);
            const int handleSlop = juce::jlimit (5, 12, getWidth() / 120);

            if (std::abs (x - startX) <= handleSlop)
            {
                draggingSegment = hit;
                draggingStart = true;
            }
            else if (std::abs (x - endX) <= handleSlop)
            {
                draggingSegment = hit;
                draggingStart = false;
            }

            repaint();
        }
    }

    void mouseDoubleClick (const juce::MouseEvent& e) override
    {
        if (segmentationPreview && getSourceWaveBounds().contains (e.getPosition()))
            resetZoomToFull();
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (scrollingZoom)
        {
            dragZoomScrollTo (e.position.x);
            return;
        }

        if (creatingSegment)
        {
            createEndSample = sampleFromX (e.position.x);
            repaint();
            return;
        }

        if (draggingSegment < 0 || ! onSegmentBoundaryMoved)
            return;

        int sample = sampleFromX (e.position.x);
        if (snapping && ! e.mods.isAltDown() && sourceData != nullptr)
            sample = extractor::snapBoundary (audioView (sourceData->audio.buffer, sampleRate), sample, juce::jmax (1, (int) (sampleRate * 0.003)));
        onSegmentBoundaryMoved (draggingSegment, draggingStart, sample);
    }

    void mouseUp (const juce::MouseEvent&) override
    {
        const bool wasCreating = creatingSegment;
        const bool wasScrolling = scrollingZoom;
        const bool wasDragging = draggingSegment >= 0;
        const int start = juce::jmin (createStartSample, createEndSample);
        const int end = juce::jmax (createStartSample, createEndSample);

        draggingSegment = -1;
        creatingSegment = false;
        scrollingZoom = false;

        if (wasCreating)
        {
            if (onSegmentCreated && end > start)
                onSegmentCreated (start, end);
            repaint();
            return;
        }

        if (wasScrolling)
        {
            repaint();
            return;
        }

        if (wasDragging && onSegmentDragFinished)
            onSegmentDragFinished();
    }

    void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
    {
        if (! segmentationPreview || originalBuffer().getNumSamples() <= 0 || ! getSourceWaveBounds().contains (e.getPosition()))
        {
            juce::Component::mouseWheelMove (e, wheel);
            return;
        }

        const float dominantDelta = std::abs (wheel.deltaX) > std::abs (wheel.deltaY) ? wheel.deltaX : wheel.deltaY;
        if (std::abs (dominantDelta) <= 0.000001f)
            return;

        if (e.mods.isShiftDown() || std::abs (wheel.deltaX) > std::abs (wheel.deltaY))
            panZoomedView (dominantDelta);
        else
            zoomAtX (e.position.x, dominantDelta);
    }

private:
    int getKeptPanelHeightForTotal (int totalHeight) const noexcept
    {
        if (! segmentationPreview)
            return juce::jmax (48, totalHeight / 2 - 4);

        constexpr int minKept = 72;
        constexpr int minSource = 140;
        const int preferred = (int) std::llround ((double) totalHeight * 0.20);
        const int maxKept = juce::jmax (minKept, totalHeight - minSource - 8);
        return juce::jlimit (minKept, maxKept, preferred);
    }

    juce::Rectangle<int> getKeptPanelBoundsFor (juce::Rectangle<int> area) const noexcept
    {
        return area.removeFromBottom (getKeptPanelHeightForTotal (area.getHeight()));
    }

    juce::Rectangle<int> getSourceWaveBounds() const
    {
        auto r = getLocalBounds().reduced (8);
        const auto kept = getKeptPanelBoundsFor (r);
        juce::ignoreUnused (kept);
        r.removeFromBottom (getKeptPanelHeightForTotal (r.getHeight()));
        r.removeFromBottom (8);
        r.removeFromTop (22);
        return r.reduced (8, 6);
    }

    int getVisibleStartSample() const noexcept
    {
        const int n = originalBuffer().getNumSamples();
        if (n <= 0)
            return 0;
        return juce::jlimit (0, juce::jmax (0, n - 1), visibleStartSample);
    }

    int getVisibleEndSample() const noexcept
    {
        const int n = originalBuffer().getNumSamples();
        if (n <= 0)
            return 0;
        return juce::jlimit (getVisibleStartSample() + 1, n, visibleEndSample);
    }

    int minimumVisibleSamples() const noexcept
    {
        const int n = originalBuffer().getNumSamples();
        if (n <= 0)
            return 0;

        const int timeFloor = sampleRate > 0.0 ? (int) std::llround (sampleRate * 0.0001) : 64;
        return juce::jlimit (1, n, juce::jmax (8, timeFloor));
    }

    void resetZoomToFull()
    {
        visibleStartSample = 0;
        visibleEndSample = juce::jmax (0, originalBuffer().getNumSamples());
        repaint();
    }

    void clampZoomRange()
    {
        const int n = originalBuffer().getNumSamples();
        if (n <= 0)
        {
            visibleStartSample = 0;
            visibleEndSample = 0;
            return;
        }

        const int minVisible = minimumVisibleSamples();
        int span = juce::jlimit (minVisible, n, visibleEndSample - visibleStartSample);
        visibleStartSample = juce::jlimit (0, juce::jmax (0, n - span), visibleStartSample);
        visibleEndSample = visibleStartSample + span;
    }

    void zoomAtX (float x, float wheelDelta)
    {
        const int n = originalBuffer().getNumSamples();
        if (n <= 0)
            return;

        const auto wave = getSourceWaveBounds();
        const int oldStart = getVisibleStartSample();
        const int oldEnd = getVisibleEndSample();
        const int oldSpan = juce::jmax (1, oldEnd - oldStart);
        const int minVisible = minimumVisibleSamples();

        const double norm = juce::jlimit (0.0, 1.0, ((double) x - (double) wave.getX()) / (double) juce::jmax (1, wave.getWidth()));
        const int anchor = juce::jlimit (0, n, (int) std::llround ((double) oldStart + norm * (double) oldSpan));
        const double factor = juce::jlimit (0.20, 5.0, std::exp ((double) -wheelDelta * 1.75));
        const int newSpan = juce::jlimit (minVisible, n, (int) std::llround ((double) oldSpan * factor));
        int newStart = (int) std::llround ((double) anchor - norm * (double) newSpan);
        newStart = juce::jlimit (0, juce::jmax (0, n - newSpan), newStart);

        visibleStartSample = newStart;
        visibleEndSample = newStart + newSpan;
        repaint();
    }

    void panZoomedView (float wheelDelta)
    {
        const int n = originalBuffer().getNumSamples();
        const int oldStart = getVisibleStartSample();
        const int oldEnd = getVisibleEndSample();
        const int span = oldEnd - oldStart;
        if (n <= 0 || span <= 0 || span >= n)
            return;

        const int step = juce::jmax (1, (int) std::llround ((double) span * 0.18 * (double) wheelDelta));
        int newStart = oldStart - step;
        newStart = juce::jlimit (0, juce::jmax (0, n - span), newStart);
        visibleStartSample = newStart;
        visibleEndSample = newStart + span;
        repaint();
    }

    void dragZoomScrollTo (float x)
    {
        const int n = originalBuffer().getNumSamples();
        const auto wave = getSourceWaveBounds();
        const int span = scrollDragVisibleSpan;
        if (n <= 0 || span <= 0 || span >= n || wave.getWidth() <= 1)
            return;

        const double samplesPerPixel = (double) span / (double) wave.getWidth();
        const int deltaSamples = (int) std::llround (((double) x - (double) scrollDragStartX) * samplesPerPixel);
        const int newStart = juce::jlimit (0, juce::jmax (0, n - span), scrollDragStartSample - deltaSamples);
        visibleStartSample = newStart;
        visibleEndSample = newStart + span;
        repaint();
    }

    int xFromSample (int sample) const
    {
        const auto wave = getSourceWaveBounds();
        const int start = getVisibleStartSample();
        const int end = getVisibleEndSample();
        const int span = juce::jmax (1, end - start);
        return wave.getX() + (int) std::llround ((double) (sample - start) * (double) wave.getWidth() / (double) span);
    }

    int sampleFromX (float x) const
    {
        const auto wave = getSourceWaveBounds();
        const int n = juce::jmax (1, originalBuffer().getNumSamples());
        const int start = getVisibleStartSample();
        const int end = getVisibleEndSample();
        const int span = juce::jmax (1, end - start);
        const double norm = ((double) x - (double) wave.getX()) / (double) juce::jmax (1, wave.getWidth());
        return juce::jlimit (0, n, (int) std::llround ((double) start + norm * (double) span));
    }

    int findSegmentAtX (float x) const
    {
        const int xi = (int) std::lround (x);
        int bodyHit = -1;
        int bestHandle = -1;
        int bestDistance = 1000000;
        const int handleSlop = juce::jlimit (5, 12, getWidth() / 120);
        const auto wave = getSourceWaveBounds();

        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (s.length() <= 0)
                continue;

            if (s.endSample < getVisibleStartSample() || s.startSample > getVisibleEndSample())
                continue;

            const int sx = xFromSample (s.startSample);
            const int ex = xFromSample (s.endSample);
            if (juce::jmax (sx, ex) < wave.getX() - handleSlop || juce::jmin (sx, ex) > wave.getRight() + handleSlop)
                continue;

            const int ds = std::abs (xi - sx);
            const int de = std::abs (xi - ex);
            const int d = juce::jmin (ds, de);
            if (d <= handleSlop && d < bestDistance)
            {
                bestDistance = d;
                bestHandle = i;
            }

            if (xi >= juce::jmin (sx, ex) && xi <= juce::jmax (sx, ex))
                bodyHit = i;
        }

        return bestHandle >= 0 ? bestHandle : bodyHit;
    }

    void drawWave (juce::Graphics& g, juce::Rectangle<int> area, const juce::AudioBuffer<float>& b, const juce::String& label, bool drawSegments)
    {
        g.setColour (juce::Colour (0xff0f1318));
        g.fillRoundedRectangle (area.toFloat(), 8.0f);
        g.setColour (juce::Colours::white.withAlpha (0.16f));
        g.drawRoundedRectangle (area.toFloat().reduced (0.5f), 8.0f, 1.0f);

        auto header = area.removeFromTop (22).reduced (8, 0);
        g.setColour (juce::Colours::white.withAlpha (0.84f));
        g.setFont (13.0f);
        juce::String text = label;
        if (drawSegments && segmentationPreview)
        {
            int enabledCount = 0;
            for (const auto& s : segments)
                if (s.enabled && s.length() > 0)
                    ++enabledCount;
            text << "  |  " << enabledCount << " segment" << (enabledCount == 1 ? "" : "s");
            if (sampleRate > 0.0 && originalBuffer().getNumSamples() > 0)
            {
                text << "  |  " << juce::String ((double) originalBuffer().getNumSamples() / sampleRate, 2) << "s full source";
                const int visibleSpan = juce::jmax (1, getVisibleEndSample() - getVisibleStartSample());
                const double zoom = (double) originalBuffer().getNumSamples() / (double) visibleSpan;
                if (zoom > 1.01)
                    text << "  |  zoom " << juce::String (zoom, 1) << "x";
            }
            text << "  |  wheel zoom, Shift+wheel/MMB-drag pan, Ctrl+drag new, Tab/Shift+Tab nav, Space play/pause, Delete remove, Ctrl+Z undo";
        }
        if (drawSegments && status.isNotEmpty())
            text << "  |  " << status;
        g.drawText (text, header, juce::Justification::centredLeft, true);

        auto wave = area.reduced (8, 6);
        const bool virtualKept = ! drawSegments && segmentationPreview;
        const int displaySamples = virtualKept ? keptSamples : b.getNumSamples();
        if (displaySamples <= 0 || sourceData == nullptr || wave.getWidth() <= 1)
        {
            g.setColour (juce::Colours::white.withAlpha (0.4f));
            g.drawText (status.isNotEmpty() ? status : juce::String ("No preview data"), wave, juce::Justification::centred, true);
            return;
        }

        if (drawSegments && segmentationPreview )
            drawSegmentOverlay (g, wave, b.getNumSamples());

        const float mid = (float) wave.getCentreY();
        const float half = (float) wave.getHeight() * 0.45f;
        juce::Path path;
        const int width = juce::jmax (1, wave.getWidth());
        const int n = displaySamples;
        const bool useZoom = drawSegments && segmentationPreview && (&b == &originalBuffer());
        const int viewStart = useZoom ? getVisibleStartSample() : 0;
        const int viewEnd = useZoom ? getVisibleEndSample() : n;
        const int viewSpan = juce::jmax (1, viewEnd - viewStart);

        for (int x = 0; x < width; ++x)
        {
            const int start = viewStart + (int) ((int64_t) x * viewSpan / width);
            const int end = viewStart + (int) ((int64_t) (x + 1) * viewSpan / width);
            extractor::Peak peak;
            if (virtualKept) peak = keptRange (start, juce::jmax (start + 1, end));
            else
            {
                const auto& cache = drawSegments ? sourceData : processedData;
                if (cache != nullptr) peak = cache->waveform.range (audioView (b, sampleRate), start, juce::jmax (start + 1, end));
            }
            const float mn = peak.low, mx = peak.high;

            const float y1 = mid - mx * half;
            const float y2 = mid - mn * half;
            path.startNewSubPath ((float) wave.getX() + (float) x, y1);
            path.lineTo ((float) wave.getX() + (float) x, y2);
        }

        g.setColour (juce::Colour (0xff7cc7ff));
        g.strokePath (path, juce::PathStrokeType (1.0f));

        if (! drawSegments && segmentationPreview && ! segments.empty())
            drawKeptSegmentOverlay (g, wave, keptSamples);
    }

    void drawKeptSegmentOverlay (juce::Graphics& g, juce::Rectangle<int> wave, int totalSamples)
    {
        if (totalSamples <= 0 || wave.getWidth() <= 1)
            return;

        int cursor = 0;
        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (! s.enabled || s.length() <= 0)
                continue;

            const int start = cursor;
            const int end = juce::jmin (totalSamples, cursor + s.length());
            cursor = end;

            if (end <= start)
                continue;

            const int x1 = wave.getX() + (int) std::llround ((double) start * (double) wave.getWidth() / (double) totalSamples);
            const int x2 = wave.getX() + (int) std::llround ((double) end * (double) wave.getWidth() / (double) totalSamples);
            const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
            const bool selected = i == selectedSegment;

            g.setColour ((selected ? juce::Colour (0xff60a5fa) : juce::Colour (0xff34d399)).withAlpha (selected ? 0.16f : 0.07f));
            g.fillRect (region);
            g.setColour ((selected ? juce::Colour (0xff93c5fd) : juce::Colour (0xffffd166)).withAlpha (0.75f));
            g.drawLine ((float) x1, (float) wave.getY(), (float) x1, (float) wave.getBottom(), selected ? 2.0f : 1.0f);
            g.drawLine ((float) x2, (float) wave.getY(), (float) x2, (float) wave.getBottom(), selected ? 2.0f : 1.0f);
        }
    }

    void drawSegmentOverlay (juce::Graphics& g, juce::Rectangle<int> wave, int totalSamples)
    {
        if (totalSamples <= 0)
            return;

        for (int i = 0; i < (int) segments.size(); ++i)
        {
            const auto& s = segments[(size_t) i];
            if (s.length() <= 0)
                continue;

            const int rawX1 = xFromSample (s.startSample);
            const int rawX2 = xFromSample (s.endSample);
            if (juce::jmax (rawX1, rawX2) < wave.getX() || juce::jmin (rawX1, rawX2) > wave.getRight())
                continue;

            const int x1 = juce::jlimit (wave.getX(), wave.getRight(), rawX1);
            const int x2 = juce::jlimit (wave.getX(), wave.getRight(), rawX2);
            const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
            const bool selected = i == selectedSegment;
            g.setColour ((selected ? juce::Colour (0xff60a5fa) : juce::Colour (0xff34d399)).withAlpha (! s.enabled ? 0.04f : (selected ? 0.22f : 0.13f)));
            g.fillRect (region);
            g.setColour ((selected ? juce::Colour (0xff93c5fd) : juce::Colour (0xffffd166)).withAlpha (0.92f));
            auto edge = [&] (int x, float quality)
            {
                if (x < wave.getX() || x > wave.getRight()) return;
                if (s.locked || quality >= 0.75f)
                    g.drawLine ((float) x, (float) wave.getY(), (float) x, (float) wave.getBottom(), s.locked ? 2.0f : 1.0f);
                else
                    for (int y = wave.getY(); y < wave.getBottom(); y += 8)
                        g.drawLine ((float) x, (float) y, (float) x, (float) juce::jmin (y + 4, wave.getBottom()), 1.0f);
            };
            edge (rawX1, s.startBoundaryScore); edge (rawX2, s.endBoundaryScore);
            if (region.getWidth() > 32)
            {
                g.setColour (juce::Colours::white.withAlpha (s.enabled ? 0.85f : 0.30f));
                g.setFont (11.0f);
                g.drawText (s.example ? "EX" : (s.locked ? (s.enabled ? "LOCK" : "REJECT") : ""), region.withHeight (16), juce::Justification::centred, true);
            }
            if (selected)
            {
                g.setColour (juce::Colour (0xff93c5fd).withAlpha (0.75f));
                g.drawRect (region, 1);
            }
        }

        for (const auto& span : reviewSpans)
        {
            const int x1 = juce::jlimit (wave.getX(), wave.getRight(), xFromSample (span.start));
            const int x2 = juce::jlimit (wave.getX(), wave.getRight(), xFromSample (span.end));
            if (x2 > x1)
            {
                g.setColour (juce::Colour (0xffffbd59).withAlpha (0.6f));
                g.fillRect (x1, wave.getBottom() - 4, x2 - x1, 3);
            }
        }
        if (focusEnd > focusStart && focusStart >= 0)
        {
            const int x1 = juce::jlimit (wave.getX(), wave.getRight(), xFromSample (focusStart));
            const int x2 = juce::jlimit (wave.getX(), wave.getRight(), xFromSample (focusEnd));
            g.setColour (juce::Colour (0xffffbd59).withAlpha (0.25f));
            g.fillRect (x1, wave.getY(), juce::jmax (0, x2 - x1), wave.getHeight());
        }
        if (cursorSample >= getVisibleStartSample() && cursorSample <= getVisibleEndSample())
        {
            g.setColour (juce::Colours::white.withAlpha (0.65f));
            g.drawVerticalLine (xFromSample (cursorSample), (float) wave.getY(), (float) wave.getBottom());
        }
        drawPendingCreatedSegment (g, wave);
    }

    void drawPendingCreatedSegment (juce::Graphics& g, juce::Rectangle<int> wave)
    {
        if (! creatingSegment)
            return;

        const int start = juce::jmin (createStartSample, createEndSample);
        const int end = juce::jmax (createStartSample, createEndSample);
        if (end <= start)
            return;

        const int rawX1 = xFromSample (start);
        const int rawX2 = xFromSample (end);
        if (juce::jmax (rawX1, rawX2) < wave.getX() || juce::jmin (rawX1, rawX2) > wave.getRight())
            return;

        const int x1 = juce::jlimit (wave.getX(), wave.getRight(), rawX1);
        const int x2 = juce::jlimit (wave.getX(), wave.getRight(), rawX2);
        const auto region = juce::Rectangle<int> (juce::jmin (x1, x2), wave.getY(), juce::jmax (1, std::abs (x2 - x1)), wave.getHeight());
        g.setColour (juce::Colour (0xffffd166).withAlpha (0.24f));
        g.fillRect (region);
        g.setColour (juce::Colour (0xffffe6a8).withAlpha (0.92f));
        g.drawRect (region, 2);
    }

    struct KeptSpan { int first = 0, last = 0, sourceStart = 0; };
    std::vector<KeptSpan> keptSpans;
    int keptSamples = 0;
    bool snapping = true;
    int cursorSample = -1, focusStart = -1, focusEnd = -1;
    std::vector<extractor::Region> reviewSpans;
    std::shared_ptr<const ImportPreviewSource> sourceData, processedData;
    const juce::AudioBuffer<float>& originalBuffer() const
    {
        static const juce::AudioBuffer<float> empty;
        return sourceData != nullptr ? sourceData->audio.buffer : empty;
    }
    const juce::AudioBuffer<float>& processedBuffer() const
    {
        static const juce::AudioBuffer<float> empty;
        return processedData != nullptr ? processedData->audio.buffer : empty;
    }
    extractor::Peak keptRange (int start, int end) const
    {
        extractor::Peak peak;
        if (sourceData == nullptr || keptSpans.empty()) return peak;
        auto it = std::lower_bound (keptSpans.begin(), keptSpans.end(), start,
                                   [] (const KeptSpan& span, int value) { return span.last <= value; });
        for (; it != keptSpans.end() && it->first < end; ++it)
        {
            const int lo = it->sourceStart + juce::jmax (0, start - it->first);
            const int hi = it->sourceStart + juce::jmin (end, it->last) - it->first;
            peak.add (sourceData->waveform.range (audioView (sourceData->audio.buffer, sampleRate), lo, hi));
        }
        return peak;
    }
    std::vector<SegmentRegion> segments;
    double sampleRate = 0.0;
    bool segmentationPreview = false;
    juce::String status;
    int visibleStartSample = 0;
    int visibleEndSample = 0;
    int selectedSegment = -1;
    int draggingSegment = -1;
    bool draggingStart = false;
    bool creatingSegment = false;
    int createStartSample = 0;
    int createEndSample = 0;
    bool scrollingZoom = false;
    float scrollDragStartX = 0.0f;
    int scrollDragStartSample = 0;
    int scrollDragVisibleSpan = 0;
    SegmentSelectCallback onSegmentSelected;
    SegmentBoundaryCallback onSegmentBoundaryMoved;
    SegmentDragFinishedCallback onSegmentDragFinished;
    SegmentCreatedCallback onSegmentCreated;
};

class ResettableSlider final : public juce::Slider
{
public:
    void setResetValue (double v)
    {
        resetValue = v;
        setDoubleClickReturnValue (true, resetValue);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (e.mods.isRightButtonDown())
        {
            setValue (resetValue, juce::sendNotificationAsync);
            return;
        }

        juce::Slider::mouseDown (e);
    }

private:
    double resetValue = 0.0;
};

class ImportPreviewComponent final : public juce::Component,
                                     private juce::Slider::Listener,
                                     private juce::Timer,
                                     private juce::KeyListener
{
public:
    using ApplyCallback = std::function<void (ImportRules)>;
    using AuditionCallback = std::function<void (juce::AudioBuffer<float>, double)>;
    using StopAuditionCallback = std::function<void()>;
    using PauseAuditionCallback = std::function<void (bool)>;

    ImportPreviewComponent (std::vector<juce::File> inputs, ImportAction actionIn, ImportRules initialRules,
                            ApplyCallback cb, AuditionCallback auditionCb = {}, StopAuditionCallback stopCb = {},
                            PauseAuditionCallback pauseCb = {}, juce::String destination = {})
        : files (initialRules.sourceBindings.empty() ? filterSupportedExistingFiles (inputs) : std::move (inputs)),
          action (actionIn), rules (std::move (initialRules)), onApply (std::move (cb)),
          onAudition (std::move (auditionCb)), onStopAudition (std::move (stopCb)), onPauseAudition (std::move (pauseCb))
    {
        defaultRules = makeDefaultRulesForAction (action);
        if (rules.sourceBindings.empty())
        {
            rules.sourceBindings.resize (files.size());
            for (size_t i = 0; i < files.size(); ++i) rules.sourceBindings[i].path = files[i].getFullPathName();
        }
        title.setText ((isSegmentationMode() ? "Sample Extractor" : "Preprocess Preview")
                       + (destination.isNotEmpty() ? "  |  " + destination : juce::String()), juce::dontSendNotification);
        title.setFont (juce::Font (17.0f, juce::Font::bold));
        title.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (title);
        addAndMakeVisible (sourceSelector);
        sourceSelector.onChange = [this]
        {
            if (updatingUi) return;
            const int next = sourceSelector.getSelectedId() - 1;
            if (next >= 0 && next < (int) files.size() && next != fileIndex)
            { reviewAcrossSources = false; changeSource (next); }
        };
        controlsViewport.setViewedComponent (&controls, false);
        controlsViewport.setScrollBarsShown (true, false);
        addAndMakeVisible (controlsViewport);
        setWantsKeyboardFocus (true);
        setMouseClickGrabsKeyboardFocus (true);

        configureSlider (sensitivity, "Fewer / more cuts", 0, 1, 0.01, rules.cutSensitivity, 0.5);
        configureSlider (gestures, "Events / whole gestures", 0, 1, 0.01, rules.wholeGestures, 0.65);
        configureSlider (tails, "Tail preservation", 0, 1, 0.01, rules.tailPreservation, 0.65);
        configureSlider (silenceDb, "Silence threshold dBFS", -90, -6, 0.5, rules.silenceThresholdDb, -50);
        configureSlider (threshold, "Relative RMS multiplier", 0, 2, 0.01, rules.silenceThresholdRatio, 0.1);
        configureSlider (minSilence, "Min quiet gap ms", 1, 5000, 1, rules.minSilenceMs, 100, 100);
        configureSlider (minSegment, "Min detected segment ms", 1, 10000, 1, rules.minSegmentMs, 25, 250);
        configureSlider (preRoll, "Pre-roll ms", 0, 500, 1, rules.preRollMs, 5, 20);
        configureSlider (postRoll, "Post-roll ms", 0, 1000, 1, rules.postRollMs, 15, 25);
        configureSlider (fade, "Fade ms", 0, 100, 0.5, rules.edgeFadeMs, 5, 10);
        configureSlider (rmsReject, "Reject below dB RMS", -120, -12, 0.5, rules.minRmsDb, -65);
        configureSlider (matchThreshold, "Example match threshold", 0.4, 0.95, 0.01, rules.matchThreshold, 0.7);
        configureSlider (segmentStart, "Start seconds", 0, 1, 0.000001, 0, 0);
        configureSlider (segmentEnd, "End seconds", 0, 1, 0.000001, 1, 1);
        setupToggle (assisted, "Assisted detector", rules.assistedExtraction, true);
        setupToggle (adaptive, "Adaptive background", rules.adaptiveBackground, true);
        setupToggle (relative, "Relative RMS gate (legacy)", rules.useRelativeRmsThreshold, true);
        setupToggle (trim, "Trim leading / trailing silence", rules.trimEdges, false);
        setupToggle (strip, "Strip internal silence", rules.stripInternalSilence, false);
        setupToggle (reject, "Reject quiet clips", rules.removeLowRms, true);
        setupToggle (normalize, "Normalize clip RMS", rules.normalizeClipsRms, false);
        setupToggle (useExamples, "Use examples", rules.useExamples, true);
        setupToggle (matchesOnly, "Only matching proposals", rules.matchesOnly, true);
        matchesOnly.setTooltip ("Nonmatching automatic regions stay visible and recoverable. Manual regions are never removed.");
        snap.setButtonText ("Snap edges (Alt bypasses)"); snap.setToggleState (true, juce::dontSendNotification);
        snap.onClick = [this] { waveform.setSnapping (snap.getToggleState()); }; controls.addAndMakeVisible (snap);
        autoplay.setButtonText ("Auto-play selected");
        autoplay.onClick = [this] { if (autoplay.getToggleState()) auditionSelection (false); }; controls.addAndMakeVisible (autoplay);

        setupButton (previous, "Prev", [this] { selectAdjacent (-1); });
        setupButton (next, "Next", [this] { selectAdjacent (1); });
        setupButton (play, "Play / pause", [this] { toggleAudition(); });
        setupButton (stop, "Stop", [this] { stopAuditionNow(); });
        setupButton (remove, "Delete / restore", [this] { deleteSelection(); });
        setupButton (reanalyse, "Re-analyse", [this]
        {
            captureUndo(); ++rules.segmentationRevision; refreshPreview (true);
        });
        reanalyse.setTooltip ("Recompute automatic proposals only. Corrected, confirmed and rejected regions remain protected.");
        setupButton (removeSource, "Remove source", [this]
        {
            captureUndo(); stopAuditionNow();
            setInputIndexDisabled (rules, fileIndex, ! isCurrentInputDisabled());
            if (isCurrentInputDisabled() && loading)
            {
                generation->fetch_add (1); pendingPreview = false; loading = false;
                status.setText ("Source removed; its saved cuts remain recoverable.", juce::dontSendNotification);
            }
            syncSourceSelector(); updateWaveform(); updateControls();
            if (! isCurrentInputDisabled() && ! source) refreshPreview (false);
        });
        setupButton (confirm, "Confirm", [this] { confirmSelection(); }, true);
        setupButton (addExample, "Add Example", [this] { addSelectedExample (false); }, true);
        setupButton (wrongEvent, "Wrong Event", [this] { addSelectedExample (true); }, true);
        setupButton (split, "Split (S)", [this] { splitSelection(); }, true);
        setupButton (merge, "Merge next (M)", [this] { mergeSelection(); }, true);
        setupButton (review, "Next Review", [this] { reviewVisitedSources = 0; nextReview(); }, true);
        setupButton (context, "Play context", [this] { auditionSelection (true); }, true);
        setupButton (profiles, "Examples / Profile", [this] { showProfileMenu(); }, true);
        setupButton (release, "Unlock selected", [this]
        {
            if (! validSelection()) return;
            captureUndo(); previewSegments[(size_t) selected].locked = false;
            storeCurrentSnapshot(); updateWaveform(); updateControls();
        }, true);
        release.setTooltip ("Make only the selected region replaceable by automatic analysis again. Undo restores its lock.");
        addAndMakeVisible (selectionInfo); selectionInfo.setJustificationType (juce::Justification::centredLeft);
        addAndMakeVisible (profileInfo); profileInfo.setJustificationType (juce::Justification::centredLeft);
        profileInfo.setFont (juce::Font (12.0f));
        addAndMakeVisible (status); status.setJustificationType (juce::Justification::centredLeft);
        status.setFont (juce::Font (12.0f));
        addAndMakeVisible (waveform);
        waveform.setWantsKeyboardFocus (true);
        waveform.setCallbacks ([this] (int index)
        {
            stopAuditionNow(); selected = index; reviewFocus.reset(); waveform.clearReviewFocus(); updateControls();
            if (autoplay.getToggleState()) auditionSelection (false);
        }, [this] (int index, bool start, int sample) { editBoundary (index, start, sample); },
        [this] { editingBoundary = false; storeCurrentSnapshot(); },
        [this] (int start, int end) { createSegment (start, end); });

        apply.setButtonText ("Apply"); cancel.setButtonText ("Cancel"); reset.setButtonText ("Reset controls");
        addAndMakeVisible (apply); addAndMakeVisible (cancel); addAndMakeVisible (reset); addAndMakeVisible (outputMode);
        outputMode.addItem ("Load separate samples", 1); outputMode.addItem ("Build texture", 2);
        outputMode.setSelectedId ((rules.segmentationOutput >= 0 ? rules.segmentationOutput == 1 : action == ImportAction::SegmentThenMegaTexture) ? 2 : 1,
                                  juce::dontSendNotification);
        outputMode.setVisible (isSegmentationMode());
        outputMode.onChange = [this] { if (! updatingUi) { captureUndo(); rules.segmentationOutput = outputMode.getSelectedId() - 1; } };
        apply.onClick = [this]
        {
            if (loading || allInputsDisabled()) return;
            stopAuditionNow(); updateRulesFromUi(); storeCurrentSnapshot();
            if (onApply) onApply (rules);
            if (auto* window = findParentComponentOfClass<juce::DialogWindow>()) window->exitModalState (1);
        };
        cancel.onClick = [this]
        { if (auto* window = findParentComponentOfClass<juce::DialogWindow>()) window->exitModalState (0); };
        reset.setTooltip ("Reset processing/detection controls, not your examples, source removals or manual cuts.");
        reset.onClick = [this]
        {
            captureUndo();
            auto replacement = defaultRules;
            replacement.sourceBindings = rules.sourceBindings;
            replacement.manualSegmentsByInput = rules.manualSegmentsByInput;
            replacement.manualSegmentSampleRates = rules.manualSegmentSampleRates;
            replacement.snapshotRevisions = rules.snapshotRevisions;
            replacement.disabledInputIndices = rules.disabledInputIndices;
            replacement.exampleProfile = rules.exampleProfile;
            replacement.useExamples = rules.useExamples;
            replacement.segmentationOutput = rules.segmentationOutput;
            replacement.outputChannels = rules.outputChannels;
            replacement.outputSampleRate = rules.outputSampleRate;
            replacement.segmentationRevision = rules.segmentationRevision + 1;
            rules = std::move (replacement); syncUiFromRules(); refreshPreview (true);
        };
        for (auto* component : segmentationControls()) component->setVisible (isSegmentationMode());
        syncSourceSelector(); syncUiFromRules();
        installKeys (*this);
        setSize (1180, 760);
        worker = std::make_unique<ImportPreviewWorker>();
        auditionWorker = std::make_unique<ImportPreviewWorker>();
        startTimerHz (20);
        refreshPreview (false);
    }

    ~ImportPreviewComponent() override
    {
        stopTimer(); closing = true;
        generation->fetch_add (1); auditionGeneration->fetch_add (1);
        stopAuditionNow();
        // Never detach: a worker may still be executing code in this plugin DLL.
        worker.reset(); auditionWorker.reset();
        removeKeys (*this);
        controlsViewport.setViewedComponent (nullptr, false);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (12);
        title.setBounds (area.removeFromTop (26));
        sourceSelector.setBounds (area.removeFromTop (28)); area.removeFromTop (6);
        auto footer = area.removeFromBottom (32);
        cancel.setBounds (footer.removeFromRight (82)); footer.removeFromRight (6);
        apply.setBounds (footer.removeFromRight (82)); footer.removeFromRight (6);
        reset.setBounds (footer.removeFromRight (112)); footer.removeFromRight (8);
        outputMode.setBounds (footer.removeFromLeft (juce::jmin (215, footer.getWidth())));
        status.setBounds (area.removeFromBottom (38)); area.removeFromBottom (4);
        const int controlWidth = getWidth() < 900 ? 240 : 288;
        controlsViewport.setBounds (area.removeFromLeft (controlWidth)); area.removeFromLeft (8);
        auto panel = juce::Rectangle<int> (0, 0, controlWidth - 18, 1400);
        auto layoutSlider = [&] (juce::Component& c) { if (c.isVisible()) { c.setBounds (panel.removeFromTop (43)); panel.removeFromTop (2); } };
        auto layoutToggle = [&] (juce::Component& c) { if (c.isVisible()) c.setBounds (panel.removeFromTop (25)); };
        layoutToggle (assisted); layoutToggle (adaptive);
        layoutSlider (sensitivity); layoutSlider (gestures); layoutSlider (tails);
        layoutToggle (useExamples); layoutToggle (matchesOnly); layoutSlider (matchThreshold);
        for (auto* c : { &silenceDb, &threshold, &minSilence, &minSegment, &preRoll, &postRoll, &fade, &rmsReject }) layoutSlider (*c);
        for (auto* c : { &relative, &trim, &strip, &reject, &normalize, &snap, &autoplay }) layoutToggle (*c);
        layoutSlider (segmentStart); layoutSlider (segmentEnd);
        auto buttonRow = [&] (juce::TextButton& a, juce::TextButton& b)
        {
            if (! a.isVisible()) return;
            auto row = panel.removeFromTop (29); panel.removeFromTop (5);
            a.setBounds (row.removeFromLeft ((row.getWidth() - 6) / 2)); row.removeFromLeft (6); b.setBounds (row);
        };
        buttonRow (previous, next); buttonRow (play, stop); buttonRow (remove, reanalyse);
        removeSource.setBounds (panel.removeFromTop (29));
        controls.setSize (controlWidth - 18, panel.getY() + 8);
        for (auto* slider : { &sensitivity, &gestures, &tails, &silenceDb, &threshold, &minSilence, &minSegment, &preRoll, &postRoll,
                              &fade, &rmsReject, &matchThreshold, &segmentStart, &segmentEnd })
            slider->setTextBoxStyle (juce::Slider::TextBoxBelow, false, controlWidth - 22, 19);
        if (isSegmentationMode())
        {
            const int columns = area.getWidth() < 520 ? 3 : 5;
            const int width = (area.getWidth() - (columns - 1) * 5) / columns;
            int column = 0; auto row = area.removeFromTop (29);
            for (auto* button : { &confirm, &addExample, &wrongEvent, &split, &merge, &review, &context, &profiles, &release })
            {
                if (column == columns) { area.removeFromTop (5); row = area.removeFromTop (29); column = 0; }
                button->setBounds (row.removeFromLeft (width)); row.removeFromLeft (5); ++column;
            }
            area.removeFromTop (4);
            profileInfo.setBounds (area.removeFromTop (22));
            selectionInfo.setBounds (area.removeFromTop (38));
        }
        waveform.setBounds (area);
    }

private:
    struct UndoState { ImportRules rules; int source = 0, selected = -1; };
    struct PreviewResult
    {
        std::shared_ptr<const ImportPreviewSource> source, processed;
        std::vector<SegmentRegion> segments;
        std::vector<extractor::Region> reviewSpans;
        juce::String error;
    };
    bool isSegmentationMode() const { return action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture; }
    bool validSelection() const { return selected >= 0 && selected < (int) previewSegments.size(); }
    bool isCurrentInputDisabled() const { return isInputIndexDisabled (rules, fileIndex); }
    bool allInputsDisabled() const
    {
        for (int i = 0; i < (int) files.size(); ++i) if (! isInputIndexDisabled (rules, i)) return false;
        return true;
    }
    std::optional<std::pair<int, int>> selectionRange() const
    {
        if (reviewFocus) return std::pair<int, int> { reviewFocus->start, reviewFocus->end };
        if (validSelection()) return std::pair<int, int> { previewSegments[(size_t) selected].startSample, previewSegments[(size_t) selected].endSample };
        return {};
    }
    std::vector<juce::Component*> segmentationControls()
    {
        return { &assisted, &adaptive, &sensitivity, &gestures, &tails, &useExamples, &matchesOnly, &matchThreshold,
                 &segmentStart, &segmentEnd, &snap, &autoplay, &previous, &next, &play, &stop, &remove, &reanalyse,
                 &confirm, &addExample, &wrongEvent, &split, &merge, &review, &context, &profiles, &release, &profileInfo, &selectionInfo };
    }
    void setupButton (juce::TextButton& button, const juce::String& text, std::function<void()> callback, bool main = false)
    {
        button.setButtonText (text); button.onClick = std::move (callback);
        if (main) addAndMakeVisible (button); else controls.addAndMakeVisible (button);
    }
    void setupToggle (juce::ToggleButton& button, const juce::String& text, bool initial, bool detection)
    {
        button.setButtonText (text); button.setToggleState (initial, juce::dontSendNotification);
        controls.addAndMakeVisible (button);
        button.onClick = [this, detection]
        {
            captureUndo(); updateRulesFromUi();
            if (detection) { ++rules.segmentationRevision; refreshPreview (true); }
            else if (isSegmentationMode()) { stopAuditionNow(); updateControls(); updateWaveform(); }
            else refreshPreview (false);
        };
    }
    void configureSlider (ResettableSlider& slider, const juce::String& label, double low, double high, double step,
                          double initial, double resetValue, double midpoint = 0.0)
    {
        slider.setRange (low, high, step); slider.setValue (initial, juce::dontSendNotification);
        slider.setSliderStyle (juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 260, 19);
        slider.setTextValueSuffix ("  " + label); slider.setTooltip (label);
        slider.setResetValue (resetValue);
        if (midpoint > low && midpoint < high) slider.setSkewFactorFromMidPoint (midpoint);
        slider.addListener (this); controls.addAndMakeVisible (slider);
    }
    void sliderValueChanged (juce::Slider* changed) override
    {
        if (updatingUi) return;
        if (! sliderUndoCaptured) { captureUndo(); sliderUndoCaptured = true; }
        if (changed == &segmentStart || changed == &segmentEnd)
        {
            if (source && validSelection())
            {
                const double sr = source->audio.sampleRate;
                editBoundary (selected, changed == &segmentStart,
                              (int) std::llround ((changed == &segmentStart ? segmentStart : segmentEnd).getValue() * sr), false);
            }
        }
        else
        {
            updateRulesFromUi();
            if (changed == &fade && isSegmentationMode()) { stopAuditionNow(); updateWaveform(); }
            else { ++rules.segmentationRevision; refreshPreview (true, 100.0); }
        }
        if (changed == nullptr || ! changed->isMouseButtonDown()) sliderUndoCaptured = false;
    }
    void sliderDragStarted (juce::Slider*) override { if (! sliderUndoCaptured) { captureUndo(); sliderUndoCaptured = true; } }
    void sliderDragEnded (juce::Slider*) override { sliderUndoCaptured = false; editingBoundary = false; }
    void updateRulesFromUi()
    {
        rules.assistedExtraction = assisted.getToggleState(); rules.adaptiveBackground = adaptive.getToggleState();
        rules.cutSensitivity = sensitivity.getValue(); rules.wholeGestures = gestures.getValue(); rules.tailPreservation = tails.getValue();
        rules.silenceThresholdDb = silenceDb.getValue(); rules.silenceThresholdRatio = (float) threshold.getValue();
        rules.minSilenceMs = minSilence.getValue(); rules.minSegmentMs = minSegment.getValue();
        rules.preRollMs = preRoll.getValue(); rules.postRollMs = postRoll.getValue(); rules.edgeFadeMs = fade.getValue();
        rules.minRmsDb = rmsReject.getValue(); rules.useRelativeRmsThreshold = relative.getToggleState();
        rules.trimEdges = trim.getToggleState(); rules.stripInternalSilence = strip.getToggleState();
        rules.removeLowRms = reject.getToggleState(); rules.normalizeClipsRms = normalize.getToggleState();
        rules.useExamples = useExamples.getToggleState(); rules.matchesOnly = matchesOnly.getToggleState(); rules.matchThreshold = matchThreshold.getValue();
        if (isSegmentationMode()) rules.segmentationOutput = outputMode.getSelectedId() - 1;
    }
    void syncUiFromRules()
    {
        updatingUi = true;
        assisted.setToggleState (rules.assistedExtraction, juce::dontSendNotification); adaptive.setToggleState (rules.adaptiveBackground, juce::dontSendNotification);
        sensitivity.setValue (rules.cutSensitivity, juce::dontSendNotification); gestures.setValue (rules.wholeGestures, juce::dontSendNotification); tails.setValue (rules.tailPreservation, juce::dontSendNotification);
        silenceDb.setValue (rules.silenceThresholdDb, juce::dontSendNotification); threshold.setValue (rules.silenceThresholdRatio, juce::dontSendNotification);
        minSilence.setValue (rules.minSilenceMs, juce::dontSendNotification); minSegment.setValue (rules.minSegmentMs, juce::dontSendNotification);
        preRoll.setValue (rules.preRollMs, juce::dontSendNotification); postRoll.setValue (rules.postRollMs, juce::dontSendNotification); fade.setValue (rules.edgeFadeMs, juce::dontSendNotification);
        rmsReject.setValue (rules.minRmsDb, juce::dontSendNotification); relative.setToggleState (rules.useRelativeRmsThreshold, juce::dontSendNotification);
        trim.setToggleState (rules.trimEdges, juce::dontSendNotification); strip.setToggleState (rules.stripInternalSilence, juce::dontSendNotification);
        reject.setToggleState (rules.removeLowRms, juce::dontSendNotification); normalize.setToggleState (rules.normalizeClipsRms, juce::dontSendNotification);
        useExamples.setToggleState (rules.useExamples, juce::dontSendNotification); matchesOnly.setToggleState (rules.matchesOnly, juce::dontSendNotification);
        matchThreshold.setValue (rules.matchThreshold, juce::dontSendNotification);
        if (rules.segmentationOutput >= 0) outputMode.setSelectedId (rules.segmentationOutput + 1, juce::dontSendNotification);
        updatingUi = false;
        syncSourceSelector(); updateControls();
    }
    void syncSourceSelector()
    {
        updatingUi = true; sourceSelector.clear (juce::dontSendNotification);
        for (int i = 0; i < (int) files.size(); ++i)
        {
            auto label = files[(size_t) i].getFileName();
            if (! files[(size_t) i].existsAsFile()) label += "  [missing]";
            if (isInputIndexDisabled (rules, i)) label += "  [removed]";
            sourceSelector.addItem (juce::String (i + 1) + ". " + label, i + 1);
        }
        fileIndex = files.empty() ? -1 : juce::jlimit (0, (int) files.size() - 1, fileIndex);
        sourceSelector.setSelectedId (fileIndex + 1, juce::dontSendNotification); updatingUi = false;
    }
    void updateControls()
    {
        const bool ready = source != nullptr && ! loading && ! isCurrentInputDisabled();
        const auto range = selectionRange(); const bool canEdit = ready && range && range->second > range->first;
        waveform.setEnabled (ready);
        segmentStart.setEnabled (canEdit && validSelection()); segmentEnd.setEnabled (canEdit && validSelection());
        for (auto* button : { &play, &confirm, &addExample, &wrongEvent, &context, &remove }) button->setEnabled (canEdit);
        play.setEnabled (canEdit && (bool) onAudition); context.setEnabled (canEdit && (bool) onAudition);
        split.setEnabled (canEdit && validSelection()); merge.setEnabled (canEdit && validSelection());
        release.setEnabled (canEdit && validSelection() && previewSegments[(size_t) selected].locked);
        previous.setEnabled (ready && ! previewSegments.empty()); next.setEnabled (ready && ! previewSegments.empty());
        stop.setEnabled (auditionActive || auditionPending); reanalyse.setEnabled (! loading && fileIndex >= 0);
        review.setEnabled (! loading && ! files.empty()); profiles.setEnabled (! loading);
        removeSource.setEnabled (fileIndex >= 0); removeSource.setButtonText (isCurrentInputDisabled() ? "Restore source" : "Remove source");
        apply.setEnabled (! loading && ! allInputsDisabled() && (previewError.isEmpty() || isCurrentInputDisabled()));
        threshold.setEnabled (! rules.assistedExtraction && relative.getToggleState());
        relative.setEnabled (! rules.assistedExtraction); adaptive.setEnabled (rules.assistedExtraction);
        silenceDb.setEnabled (! rules.assistedExtraction || ! rules.adaptiveBackground);
        rmsReject.setEnabled (rules.removeLowRms);
        trim.setEnabled (! isSegmentationMode()); strip.setEnabled (! isSegmentationMode());
        matchThreshold.setEnabled (rules.useExamples); matchesOnly.setEnabled (rules.useExamples);
        for (auto* slider : { &sensitivity, &gestures, &tails }) slider->setEnabled (rules.assistedExtraction);
        play.setButtonText (auditionActive ? (auditionPaused ? "Resume" : "Pause") : (auditionPending ? "Preparing..." : "Play"));
        const int positives = rules.exampleProfile.positives(); const int negatives = (int) rules.exampleProfile.examples.size() - positives;
        profileInfo.setText ("Examples: " + juce::String (positives) + " positive / " + juce::String (negatives) + " negative"
                             + (rules.useExamples ? "  |  enabled" : "  |  disabled") + "  |  dashed edges need review; solid = locked/strong",
                             juce::dontSendNotification);
        if (canEdit)
        {
            const double sr = source->audio.sampleRate;
            updatingUi = true;
            segmentStart.setRange (0.0, source->audio.buffer.getNumSamples() / sr, 1.0 / sr);
            segmentEnd.setRange (0.0, source->audio.buffer.getNumSamples() / sr, 1.0 / sr);
            segmentStart.setValue (range->first / sr, juce::dontSendNotification); segmentEnd.setValue (range->second / sr, juce::dontSendNotification);
            updatingUi = false;
            juce::String text = juce::String (range->first / sr, 3) + " - " + juce::String (range->second / sr, 3) + " s";
            if (reviewFocus) text += "  |  " + juce::String (extractor::reasonName (reviewFocus->reason)) + " (not yet kept)";
            else
            {
                const auto& s = previewSegments[(size_t) selected];
                text += s.locked ? "  |  LOCKED" : "  |  " + juce::String (extractor::reasonName (s.reason));
                text += "\nEvent: " + juce::String (extractor::qualityName (s.eventScore)) + "   Start: " + extractor::qualityName (s.startBoundaryScore)
                      + "   End: " + extractor::qualityName (s.endBoundaryScore) + (s.enabled ? "" : "  [rejected; Confirm restores]");
            }
            selectionInfo.setText (text, juce::dontSendNotification);
        }
        else selectionInfo.setText (loading ? "Analysing..." : "Select a region or use Ctrl+drag to mark a complete event.", juce::dontSendNotification);
        waveform.setSelectedSegment (selected);
    }
    void storeCurrentSnapshot()
    {
        if (! loading && ! pendingPreview && source && isSegmentationMode() && fileIndex >= 0
            && fileIndex < (int) files.size() && source->fingerprint.path == files[(size_t) fileIndex].getFullPathName())
            storeSegmentSnapshot (rules, fileIndex, previewSegments, source->audio.sampleRate);
    }
    void captureUndo()
    {
        storeCurrentSnapshot(); undo.push_back ({ rules, fileIndex, selected });
        if (undo.size() > 64) undo.erase (undo.begin()); redo.clear();
    }
    void restoreHistory (bool forward)
    {
        auto& from = forward ? redo : undo; auto& to = forward ? undo : redo;
        if (from.empty()) return;
        stopAuditionNow(); to.push_back ({ rules, fileIndex, selected });
        auto state = from.back(); from.pop_back(); rules = std::move (state.rules); fileIndex = state.source; selected = state.selected;
        reviewFocus.reset(); reviewCursorSample = -1; syncUiFromRules(); refreshPreview (false);
    }
    void changeSource (int index)
    {
        stopAuditionNow(); storeCurrentSnapshot(); fileIndex = index; selected = -1; reviewCursorSample = -1; reviewFocus.reset();
        source.reset(); processed.reset(); previewSegments.clear(); reviewSpans.clear(); updateWaveform(); syncSourceSelector(); refreshPreview (false);
    }
    void refreshPreview (bool force, double debounceMs = 0.0)
    {
        generation->fetch_add (1); // Invalidates a worker result before debounce elapses.
        stopAuditionNow(); loading = true; forcePending = force; pendingPreview = true;
        previewAtMs = juce::Time::getMillisecondCounterHiRes() + debounceMs;
        status.setText (source ? "Updating proposals; manual decisions remain protected..." : "Reading source and building analysis / waveform caches...", juce::dontSendNotification);
        updateControls();
        if (debounceMs <= 0.0) startPreviewJob();
    }
    void startPreviewJob()
    {
        if (! pendingPreview || ! worker) return;
        pendingPreview = false;
        if (fileIndex < 0 || fileIndex >= (int) files.size())
        { loading = false; previewError = "No input files."; status.setText (previewError, juce::dontSendNotification); updateControls(); return; }
        const auto id = generation->load(); auto token = generation; const int index = fileIndex;
        const auto file = files[(size_t) index]; const auto snapshot = rules; const bool segmentation = isSegmentationMode();
        const bool force = forcePending;
        const auto cached = source;
        juce::Component::SafePointer<ImportPreviewComponent> safe (this);
        worker->submit ([safe, id, token, index, file, snapshot, segmentation, force, cached]
        {
            const extractor::Cancel cancelJob = [token, id] { return token->load() != id; };
            if (cancelJob()) return;
            auto result = std::make_shared<PreviewResult>();
            try
            {
                const auto actual = fingerprintForFile (file);
                if (! file.existsAsFile()) result->error = "Missing source: " + file.getFullPathName() + ". Restore the file or remove this source.";
                else if (index < (int) snapshot.sourceBindings.size() && ! sourceFingerprintMatches (snapshot.sourceBindings[(size_t) index], actual))
                    result->error = "Source changed since its cuts were saved. Start a fresh import for this changed file.";
                else if (cached && sourceFingerprintMatches (cached->fingerprint, actual))
                    result->source = cached;
                else
                {
                    auto data = std::make_shared<ImportPreviewSource>();
                    data->fingerprint = actual;
                    auto read = readAudioFile (file, segmentation ? 0 : snapshot.outputChannels, segmentation ? 0.0 : snapshot.outputSampleRate,
                                               segmentation ? 0.0 : snapshot.previewSeconds, result->error, cancelJob);
                    if (read)
                    {
                        data->audio = std::move (*read);
                        if (segmentation) data->features = extractor::analyse (audioView (data->audio.buffer, data->audio.sampleRate), cancelJob);
                        if (data->waveform.build (audioView (data->audio.buffer, data->audio.sampleRate), cancelJob)) result->source = std::move (data);
                    }
                }
                if (result->source && ! cancelJob())
                {
                    const auto& src = *result->source;
                    if (segmentation)
                    {
                        result->segments = segmentsForInput (snapshot, index, src.audio.buffer, src.audio.sampleRate, &src.features, cancelJob, &result->reviewSpans, force);
                        // Exact saved proposals do not need regeneration, but the
                        // review queue still needs to expose possible missed spans.
                        if (! force && snapshot.assistedExtraction && result->reviewSpans.empty())
                        {
                            result->reviewSpans = extractor::detect (src.features, extractionSettings (snapshot), cancelJob).reviewSpans;
                            result->reviewSpans.erase (std::remove_if (result->reviewSpans.begin(), result->reviewSpans.end(), [&] (const extractor::Region& r)
                            { for (const auto& s : result->segments) if (s.startSample < r.end && s.endSample > r.start) return true; return false; }), result->reviewSpans.end());
                        }
                    }
                    else if (! isInputIndexDisabled (snapshot, index))
                    {
                        auto output = std::make_shared<ImportPreviewSource>(); output->audio.sampleRate = src.audio.sampleRate;
                        output->audio.buffer = processBufferByRules (src.audio.buffer, src.audio.sampleRate, snapshot);
                        if (output->waveform.build (audioView (output->audio.buffer, output->audio.sampleRate), cancelJob)) result->processed = std::move (output);
                    }
                }
            }
            catch (const std::exception& error) { result->error = "Preview failed: " + juce::String (error.what()); result->source.reset(); }
            catch (...) { result->error = "Preview failed while reading or analysing audio."; result->source.reset(); }
            if (cancelJob()) return;
            juce::MessageManager::callAsync ([safe, id, index, result]
            {
                if (safe != nullptr && ! safe->closing && safe->generation->load() == id && safe->fileIndex == index)
                    safe->acceptPreview (*result);
            });
        });
    }
    void acceptPreview (const PreviewResult& result)
    {
        loading = false; source = result.source; processed = result.processed; previewSegments = result.segments; reviewSpans = result.reviewSpans;
        previewError = result.error;
        if (source)
        {
            if ((int) rules.sourceBindings.size() <= fileIndex) rules.sourceBindings.resize ((size_t) fileIndex + 1);
            rules.sourceBindings[(size_t) fileIndex] = source->fingerprint;
            storeCurrentSnapshot();
            int kept = 0, locked = 0;
            for (const auto& s : previewSegments) { if (s.enabled) ++kept; if (s.locked) ++locked; }
            status.setText (files[(size_t) fileIndex].getFileName() + "  |  " + juce::String (source->audio.buffer.getNumSamples() / source->audio.sampleRate, 2)
                            + " s  |  " + juce::String (kept) + " kept / " + juce::String (locked) + " protected  |  "
                            + juce::String ((int) reviewSpans.size()) + " possible missed spans", juce::dontSendNotification);
            if (! validSelection())
            {
                selected = -1;
                for (int i = 0; i < (int) previewSegments.size(); ++i) if (previewSegments[(size_t) i].enabled) { selected = i; break; }
            }
        }
        else { selected = -1; status.setText (previewError, juce::dontSendNotification); }
        updateWaveform(); updateControls();
        if (reviewAcrossSources) nextReview();
        else if (autoplay.getToggleState()) auditionSelection (false);
    }
    void updateWaveform()
    {
        auto visible = previewSegments;
        if (isCurrentInputDisabled()) for (auto& s : visible) s.enabled = false;
        waveform.setSource (source, processed, std::move (visible), isSegmentationMode(), previewError, reviewSpans);
        waveform.setSelectedSegment (selected);
    }
    void edited()
    {
        generation->fetch_add (1); pendingPreview = false; loading = false; stopAuditionNow();
        storeCurrentSnapshot(); updateWaveform(); updateControls(); updateStatusSummary();
    }
    void updateStatusSummary()
    {
        if (! source || loading || fileIndex < 0 || fileIndex >= (int) files.size()) return;
        int kept = 0, locked = 0;
        for (const auto& s : previewSegments) { if (s.enabled && s.length() > 0) ++kept; if (s.locked) ++locked; }
        status.setText (files[(size_t) fileIndex].getFileName() + "  |  " + juce::String (source->audio.buffer.getNumSamples() / source->audio.sampleRate, 2)
                        + " s  |  " + juce::String (kept) + " kept / " + juce::String (locked) + " protected  |  "
                        + juce::String ((int) reviewSpans.size()) + " possible missed spans", juce::dontSendNotification);
    }
    void selectAdjacent (int direction)
    {
        if (previewSegments.empty() || loading) return;
        stopAuditionNow(); reviewFocus.reset(); waveform.clearReviewFocus(); int index = selected;
        for (int guard = 0; guard < (int) previewSegments.size(); ++guard)
        {
            index = (index + direction + (int) previewSegments.size()) % (int) previewSegments.size();
            if (previewSegments[(size_t) index].enabled && previewSegments[(size_t) index].length() > 0)
            {
                selected = index; updateControls(); waveform.reveal (previewSegments[(size_t) index].startSample, previewSegments[(size_t) index].endSample);
                if (autoplay.getToggleState()) auditionSelection (false); return;
            }
        }
    }
    void createSegment (int start, int end, bool enabled = true, bool recordUndo = true)
    {
        if (! source || loading || isCurrentInputDisabled()) return;
        const int n = source->audio.buffer.getNumSamples(); start = juce::jlimit (0, n, start); end = juce::jlimit (start, n, end);
        if (end <= start) return;
        if (recordUndo) captureUndo();
        std::vector<SegmentRegion> updated;
        for (const auto& old : previewSegments)
        {
            if (old.endSample <= start || old.startSample >= end) { updated.push_back (old); continue; }
            if (old.startSample < start) { auto left = old; left.endSample = start; left.locked = true; left.example = false; updated.push_back (left); }
            if (old.endSample > end) { auto right = old; right.startSample = end; right.locked = true; right.example = false; updated.push_back (right); }
        }
        SegmentRegion created; created.startSample = start; created.endSample = end; created.enabled = enabled; created.locked = true; created.reason = extractor::Reason::Manual;
        updated.push_back (created);
        std::stable_sort (updated.begin(), updated.end(), [] (const SegmentRegion& a, const SegmentRegion& b) { return a.startSample < b.startSample; });
        previewSegments = std::move (updated); selected = -1;
        for (int i = 0; i < (int) previewSegments.size(); ++i)
            if (previewSegments[(size_t) i].startSample == start && previewSegments[(size_t) i].endSample == end) { selected = i; break; }
        reviewFocus.reset();
        reviewSpans.erase (std::remove_if (reviewSpans.begin(), reviewSpans.end(), [=] (const extractor::Region& r) { return r.start < end && r.end > start; }), reviewSpans.end());
        edited(); if (enabled && autoplay.getToggleState()) auditionSelection (false);
    }
    void editBoundary (int index, bool startEdge, int sample, bool recordUndo = true)
    {
        if (! source || loading || index < 0 || index >= (int) previewSegments.size()) return;
        if (recordUndo && ! editingBoundary) { captureUndo(); editingBoundary = true; }
        selected = index; auto& region = previewSegments[(size_t) index];
        if (startEdge) region.startSample = juce::jlimit (0, juce::jmax (0, region.endSample - 1), sample);
        else region.endSample = juce::jlimit (region.startSample + 1, source->audio.buffer.getNumSamples(), sample);
        region.locked = true; region.example = false; region.reason = extractor::Reason::Manual;
        // Repair all affected neighbours, not just an adjacent vector position
        // (deleted entries can lie between two enabled neighbours).
        for (int i = 0; i < (int) previewSegments.size(); ++i)
        {
            if (i == index) continue;
            auto& other = previewSegments[(size_t) i];
            if (other.endSample <= region.startSample || other.startSample >= region.endSample) continue;
            other.locked = true; other.example = false;
            if (i < index) other.endSample = juce::jmax (other.startSample, region.startSample);
            else other.startSample = juce::jmin (other.endSample, region.endSample);
            if (other.length() <= 0) other.enabled = false;
        }
        edited();
    }
    void confirmSelection()
    {
        if (loading || ! source) return;
        if (reviewFocus) { createSegment (reviewFocus->start, reviewFocus->end); return; }
        if (! validSelection()) return;
        captureUndo(); auto& s = previewSegments[(size_t) selected]; s.locked = true; s.enabled = true;
        edited(); selectAdjacent (1);
    }
    void deleteSelection()
    {
        if (loading || ! source) return;
        if (reviewFocus) { createSegment (reviewFocus->start, reviewFocus->end, false); return; }
        if (! validSelection()) return;
        captureUndo(); auto& s = previewSegments[(size_t) selected]; s.enabled = ! s.enabled; s.locked = true;
        const bool removed = ! s.enabled; edited(); if (removed) selectAdjacent (1);
    }
    void splitSelection()
    {
        if (! validSelection() || loading) return;
        auto old = previewSegments[(size_t) selected]; if (old.length() < 2) return;
        int cut = waveform.getCursorSample(); if (cut <= old.startSample || cut >= old.endSample) cut = old.startSample + old.length() / 2;
        captureUndo(); auto left = old, right = old;
        left.endSample = cut; right.startSample = cut; left.locked = right.locked = true;
        left.example = right.example = false; left.reason = right.reason = extractor::Reason::Manual;
        previewSegments[(size_t) selected] = left; previewSegments.insert (previewSegments.begin() + selected + 1, right); edited();
    }
    void mergeSelection()
    {
        if (! validSelection() || loading) return;
        int nextIndex = selected + 1;
        while (nextIndex < (int) previewSegments.size() && ! previewSegments[(size_t) nextIndex].enabled) ++nextIndex;
        if (nextIndex >= (int) previewSegments.size()) return;
        createSegment (previewSegments[(size_t) selected].startSample, previewSegments[(size_t) nextIndex].endSample);
    }
    void addSelectedExample (bool negative)
    {
        const auto range = selectionRange();
        if (! source || loading || ! range) return;
        const double duration = (range->second - range->first) / source->audio.sampleRate;
        if (duration < 0.04 || duration > 60.0)
        { status.setText ("Examples must contain 40 ms to 60 s of audio. The selection itself is unchanged.", juce::dontSendNotification); return; }
        if (rules.exampleProfile.examples.size() >= extractor::maxExamples)
        { status.setText ("This profile already contains 32 examples. Remove one through Examples / Profile first.", juce::dontSendNotification); return; }
        auto example = extractor::makeExample (source->features, range->first, range->second, negative);
        if (example.frames.empty())
        { status.setText ("Select an audible event rather than digital silence.", juce::dontSendNotification); return; }
        captureUndo();
        example.name = (files[(size_t) fileIndex].getFileNameWithoutExtension() + " @ " + juce::String (range->first / source->audio.sampleRate, 3) + "s").toStdString();
        example.sourceId = source->fingerprint.path.toStdString(); rules.exampleProfile.examples.push_back (std::move (example));
        if (reviewFocus) createSegment (range->first, range->second, ! negative, false);
        if (validSelection())
        {
            auto& s = previewSegments[(size_t) selected]; s.locked = true; s.enabled = ! negative; s.example = true;
        }
        rules.useExamples = true; storeCurrentSnapshot(); ++rules.segmentationRevision;
        syncUiFromRules(); refreshPreview (true);
    }
    void nextReview()
    {
        if (loading || files.empty()) return;
        stopAuditionNow();
        struct Item { int start, end, index; extractor::Region span; };
        std::vector<Item> items;
        if (! isCurrentInputDisabled())
        {
            for (int i = 0; i < (int) previewSegments.size(); ++i)
            {
                const auto& s = previewSegments[(size_t) i];
                if (! s.locked && s.length() > 0 && (std::min ({ s.eventScore, s.startBoundaryScore, s.endBoundaryScore }) < 0.75f || ! s.enabled))
                    items.push_back ({ s.startSample, s.endSample, i, {} });
            }
            for (const auto& span : reviewSpans) items.push_back ({ span.start, span.end, -1, span });
        }
        std::stable_sort (items.begin(), items.end(), [] (const Item& a, const Item& b) { return a.start < b.start; });
        for (const auto& item : items)
        {
            if (item.start <= reviewCursorSample) continue;
            reviewCursorSample = item.start; selected = item.index; reviewAcrossSources = false;
            if (item.index < 0) reviewFocus = item.span; else reviewFocus.reset();
            waveform.setReviewFocus (item.start, item.end); updateControls();
            if (autoplay.getToggleState()) auditionSelection (false); return;
        }
        if (++reviewVisitedSources >= (int) files.size())
        {
            reviewAcrossSources = false; reviewCursorSample = -1;
            status.setText ("Review pass complete. Protected decisions were skipped; Next Review starts another pass.", juce::dontSendNotification); return;
        }
        reviewAcrossSources = true; changeSource ((fileIndex + 1) % (int) files.size());
    }
    void showProfileMenu()
    {
        juce::PopupMenu menu;
        menu.addItem (1, "Save example profile...", ! rules.exampleProfile.examples.empty());
        menu.addItem (2, "Load example profile...");
        menu.addItem (3, "Clear examples (keep manual edits)", ! rules.exampleProfile.examples.empty());
        if (! rules.exampleProfile.examples.empty()) menu.addSeparator();
        for (size_t i = 0; i < rules.exampleProfile.examples.size(); ++i)
        {
            const auto& e = rules.exampleProfile.examples[i];
            menu.addItem (100 + (int) i, "Remove " + juce::String (e.negative ? "negative: " : "positive: ") + juce::String::fromUTF8 (e.name.c_str()));
        }
        juce::Component::SafePointer<ImportPreviewComponent> safe (this);
        menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&profiles), [safe] (int result)
        {
            if (safe == nullptr || result == 0) return;
            if (result == 1 || result == 2) { safe->chooseProfileFile (result == 1); return; }
            safe->captureUndo();
            auto& examples = safe->rules.exampleProfile.examples;
            if (result == 3) examples.clear();
            else if (result >= 100 && result - 100 < (int) examples.size()) examples.erase (examples.begin() + result - 100);
            if (safe->rules.exampleProfile.positives() == 0) safe->rules.useExamples = false;
            ++safe->rules.segmentationRevision; safe->syncUiFromRules(); safe->refreshPreview (true);
        });
    }
    void chooseProfileFile (bool saving)
    {
        chooser = std::make_unique<juce::FileChooser> (saving ? "Save extraction example profile" : "Load extraction example profile",
                                                      juce::File(), "*.zaextract.xml;*.xml");
        const auto flags = saving ? (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting)
                                  : (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles);
        juce::Component::SafePointer<ImportPreviewComponent> safe (this);
        chooser->launchAsync (flags, [safe, saving] (const juce::FileChooser& selectedFile)
        {
            if (safe == nullptr) return;
            auto file = selectedFile.getResult(); if (file == juce::File()) return;
            if (saving)
            {
                if (! file.getFileName().endsWithIgnoreCase (".xml")) file = file.getSiblingFile (file.getFileName() + ".zaextract.xml");
                auto profile = safe->rules.exampleProfile; profile.name = file.getFileNameWithoutExtension().toStdString();
                auto xml = extractionProfileToValueTree (profile).createXml();
                if (xml == nullptr || ! file.replaceWithText (xml->toString())) safe->status.setText ("Could not save the example profile.", juce::dontSendNotification);
                else safe->status.setText ("Saved profile: " + file.getFileName(), juce::dontSendNotification);
            }
            else
            {
                if (file.getSize() > 4 * 1024 * 1024) { safe->status.setText ("Profile exceeds the 4 MiB safety limit.", juce::dontSendNotification); return; }
                auto xml = juce::parseXML (file); extractor::Profile profile;
                if (xml == nullptr || ! extractionProfileFromValueTree (juce::ValueTree::fromXml (*xml), profile))
                { safe->status.setText ("Invalid profile or incompatible analysis version. Existing examples and edits were not changed.", juce::dontSendNotification); return; }
                safe->captureUndo(); safe->rules.exampleProfile = std::move (profile); safe->rules.useExamples = safe->rules.exampleProfile.positives() > 0;
                ++safe->rules.segmentationRevision; safe->syncUiFromRules(); safe->refreshPreview (true);
            }
        });
    }
    void auditionSelection (bool withContext)
    {
        const auto range = selectionRange();
        if (! source || loading || isCurrentInputDisabled() || ! range || ! onAudition || ! auditionWorker) return;
        stopAuditionNow();
        const auto src = source; const auto renderRules = rules;
        const int padding = withContext ? (int) std::llround (src->audio.sampleRate * 0.2) : 0;
        const int start = juce::jmax (0, range->first - padding), end = juce::jmin (src->audio.buffer.getNumSamples(), range->second + padding);
        if (end <= start) return;
        const auto id = auditionGeneration->fetch_add (1) + 1; const auto token = auditionGeneration;
        auditionPending = true; updateControls();
        juce::Component::SafePointer<ImportPreviewComponent> safe (this);
        auditionWorker->submit ([safe, src, renderRules, withContext, start, end, id, token]
        {
            if (token->load() != id) return;
            auto clip = std::make_shared<juce::AudioBuffer<float>>(); juce::String error;
            try
            {
                *clip = copyRange (src->audio.buffer, start, end);
                applyEdgeFades (*clip, src->audio.sampleRate, withContext ? 3.0 : renderRules.edgeFadeMs);
                if (! withContext && renderRules.normalizeClipsRms)
                {
                    const double rms = computeRmsLinear (*clip);
                    if (rms > 1.0e-9) clip->applyGain ((float) (dbToLinear (renderRules.clipTargetRmsDb) / rms));
                }
            }
            catch (...) { error = "Not enough memory to prepare this audition."; }
            if (token->load() != id) return;
            juce::MessageManager::callAsync ([safe, clip, sr = src->audio.sampleRate, id, error]
            {
                if (safe == nullptr || safe->closing || safe->auditionGeneration->load() != id) return;
                safe->auditionPending = false;
                if (error.isNotEmpty()) { safe->status.setText (error, juce::dontSendNotification); safe->updateControls(); return; }
                safe->auditionRemainingMs = 1000.0 * clip->getNumSamples() / sr;
                safe->onAudition (std::move (*clip), sr);
                safe->auditionActive = true; safe->auditionPaused = false;
                safe->auditionEndMs = juce::Time::getMillisecondCounterHiRes() + safe->auditionRemainingMs;
                safe->updateControls();
            });
        });
    }
    void toggleAudition()
    {
        if (! auditionActive) { auditionSelection (false); return; }
        if (! onPauseAudition) { stopAuditionNow(); return; } // Never display a fake paused state.
        if (auditionPaused) { auditionPaused = false; auditionEndMs = juce::Time::getMillisecondCounterHiRes() + auditionRemainingMs; }
        else { auditionRemainingMs = juce::jmax (1.0, auditionEndMs - juce::Time::getMillisecondCounterHiRes()); auditionPaused = true; }
        onPauseAudition (auditionPaused); updateControls();
    }
    void stopAuditionNow()
    {
        auditionGeneration->fetch_add (1); auditionPending = false;
        if (onStopAudition) onStopAudition();
        auditionActive = auditionPaused = false; auditionEndMs = auditionRemainingMs = 0.0;
        if (! closing) updateControls();
    }
    void timerCallback() override
    {
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (pendingPreview && now >= previewAtMs) startPreviewJob();
        if (auditionActive && ! auditionPaused && now >= auditionEndMs)
        { auditionActive = false; auditionRemainingMs = 0.0; updateControls(); }
    }
    void installKeys (juce::Component& component)
    {
        component.addKeyListener (this);
        for (int i = 0; i < component.getNumChildComponents(); ++i) if (auto* child = component.getChildComponent (i)) installKeys (*child);
    }
    void removeKeys (juce::Component& component)
    {
        component.removeKeyListener (this);
        for (int i = 0; i < component.getNumChildComponents(); ++i) if (auto* child = component.getChildComponent (i)) removeKeys (*child);
    }
    bool shortcut (const juce::KeyPress& key)
    {
        if (! isSegmentationMode()) return false;
        if (dynamic_cast<juce::TextEditor*> (juce::Component::getCurrentlyFocusedComponent()) != nullptr) return false;
        const auto mods = key.getModifiers(); const int code = key.getKeyCode(); const bool control = mods.isCtrlDown() || mods.isCommandDown();
        if (control && (code == 'Z' || code == 'z')) { restoreHistory (mods.isShiftDown()); return true; }
        if (loading) return false;
        if (code == juce::KeyPress::spaceKey) { toggleAudition(); return true; }
        if (code == juce::KeyPress::tabKey && ! control) { selectAdjacent (mods.isShiftDown() ? -1 : 1); return true; }
        if (code == juce::KeyPress::deleteKey || code == juce::KeyPress::backspaceKey) { deleteSelection(); return true; }
        if (! control && (code == 'S' || code == 's')) { splitSelection(); return true; }
        if (! control && (code == 'M' || code == 'm')) { mergeSelection(); return true; }
        if (! control && (code == 'E' || code == 'e')) { addSelectedExample (false); return true; }
        if (! control && (code == 'R' || code == 'r')) { reviewVisitedSources = 0; nextReview(); return true; }
        if (code == juce::KeyPress::returnKey && ! control) { confirmSelection(); return true; }
        return false;
    }
    bool keyPressed (const juce::KeyPress& key) override { return shortcut (key); }
    bool keyPressed (const juce::KeyPress& key, juce::Component*) override { return shortcut (key); }

    std::vector<juce::File> files;
    ImportAction action;
    ImportRules rules, defaultRules;
    ApplyCallback onApply;
    AuditionCallback onAudition;
    StopAuditionCallback onStopAudition;
    PauseAuditionCallback onPauseAudition;
    juce::Label title, selectionInfo, profileInfo, status;
    juce::ComboBox sourceSelector, outputMode;
    juce::Viewport controlsViewport;
    juce::Component controls;
    ResettableSlider sensitivity, gestures, tails, silenceDb, threshold, minSilence, minSegment, preRoll, postRoll, fade, rmsReject, matchThreshold, segmentStart, segmentEnd;
    juce::ToggleButton assisted, adaptive, relative, trim, strip, reject, normalize, useExamples, matchesOnly, snap, autoplay;
    juce::TextButton previous, next, play, stop, remove, reanalyse, removeSource, confirm, addExample, wrongEvent, split, merge, review, context, profiles, release, apply, cancel, reset;
    WaveformPreview waveform;
    std::unique_ptr<juce::FileChooser> chooser;
    std::shared_ptr<const ImportPreviewSource> source, processed;
    std::vector<SegmentRegion> previewSegments;
    std::vector<extractor::Region> reviewSpans;
    std::optional<extractor::Region> reviewFocus;
    std::vector<UndoState> undo, redo;
    int fileIndex = 0, selected = -1, reviewCursorSample = -1, reviewVisitedSources = 0;
    bool updatingUi = false, loading = false, closing = false, editingBoundary = false, sliderUndoCaptured = false;
    bool pendingPreview = false, forcePending = false, reviewAcrossSources = false;
    bool auditionActive = false, auditionPaused = false, auditionPending = false;
    double previewAtMs = 0.0, auditionEndMs = 0.0, auditionRemainingMs = 0.0;
    juce::String previewError;
    std::shared_ptr<std::atomic<uint64_t>> generation = std::make_shared<std::atomic<uint64_t>> (0);
    std::shared_ptr<std::atomic<uint64_t>> auditionGeneration = std::make_shared<std::atomic<uint64_t>> (0);
    std::unique_ptr<ImportPreviewWorker> worker, auditionWorker;
};

static inline juce::Rectangle<int> importPreviewUsableDisplayAreaFor (juce::Component& parent)
{
    const auto& displays = juce::Desktop::getInstance().getDisplays();
    if (auto* display = displays.getDisplayForRect (parent.getScreenBounds(), false)) return display->userArea;
    if (auto* display = displays.getPrimaryDisplay()) return display->userArea;
    return { 0, 0, 1280, 800 };
}
static inline juce::Rectangle<int> importPreviewDialogBoundsFor (juce::Component& parent, int desiredW, int desiredH)
{
    auto area = importPreviewUsableDisplayAreaFor (parent).reduced (18);
    if (area.isEmpty()) area = { 18, 18, 1244, 764 };
    const int w = juce::jlimit (juce::jmin (720, area.getWidth()), area.getWidth(), desiredW);
    const int h = juce::jlimit (juce::jmin (520, area.getHeight()), area.getHeight(), desiredH);
    auto centre = parent.getScreenBounds().getCentre(); if (! area.contains (centre)) centre = area.getCentre();
    return juce::Rectangle<int> (w, h).withCentre (centre).constrainedWithin (area);
}
static inline void showImportPreviewDialog (juce::Component& parent, std::vector<juce::File> files, ImportAction action, ImportRules rules,
                                           ImportPreviewComponent::ApplyCallback onApply,
                                           ImportPreviewComponent::AuditionCallback onAudition = {},
                                           ImportPreviewComponent::StopAuditionCallback onStopAudition = {},
                                           ImportPreviewComponent::PauseAuditionCallback onPauseAudition = {},
                                           juce::String destination = {})
{
    const bool segmentation = action == ImportAction::SegmentLongFile || action == ImportAction::SegmentThenMegaTexture;
    const auto bounds = importPreviewDialogBoundsFor (parent, segmentation ? 1180 : 1040, segmentation ? 760 : 700);
    auto* content = new ImportPreviewComponent (std::move (files), action, std::move (rules), std::move (onApply), std::move (onAudition), std::move (onStopAudition), std::move (onPauseAudition), std::move (destination));
    content->setSize (bounds.getWidth(), bounds.getHeight());
    juce::DialogWindow::LaunchOptions opts;
    opts.dialogTitle = segmentation ? "Sample Extractor" : "Import / Preprocess";
    opts.dialogBackgroundColour = juce::Colour (0xff20272d); opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true; opts.resizable = true; opts.content.setOwned (content);
    if (auto* window = opts.launchAsync())
    {
        window->setResizable (true, true);
        const auto screen = importPreviewUsableDisplayAreaFor (parent).reduced (18);
        window->setResizeLimits (juce::jmin (720, bounds.getWidth()), juce::jmin (520, bounds.getHeight()), screen.getWidth(), screen.getHeight());
        window->setBounds (bounds);
    }
}
} // namespace za::fileimport
