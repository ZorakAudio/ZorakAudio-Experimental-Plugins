// SPDX-License-Identifier: Zlib
// Native worker file backend, adapted from YSFXGfxInterpreter.h.
  struct FileData
  {
    enum class ReadMode : uint8_t { none, numeric, string };

    bool textMode = false; // Native file_text(): only .txt is text mode.
    int channels = 0;
    int64_t frames = 0;
    double sampleRate = 0.0;
    std::vector<double> items;          // Audio interleaved, or parsed text tokens.
    std::vector<juce::String> lines;    // file_string() view for non-media files.
  };

  struct FileHandle
  {
    std::vector<juce::File> files;
    std::unique_ptr<FileData> data;
    int selectedFile = 0;
    int64_t cursor = 0;
    int64_t lineCursor = 0;
    FileData::ReadMode readMode = FileData::ReadMode::none;
  };

  juce::AudioFormatManager fileFormatManager;
  std::vector<FileHandle> fileHandles;
  std::vector<int> freeFileHandles;

std::function<std::vector<juce::File>(int,const juce::String&)> fileResolver;
std::array<juce::String,128> fileResourceNames{};
std::array<bool,128> fileResourceDeclared{};
static constexpr uint32_t kRqsrFourCC=(uint32_t('r')<<24)|(uint32_t('q')<<16)|(uint32_t('s')<<8)|uint32_t('r');
static bool writeUtf8ToStringArgument(void* opaque,Reference slot,const juce::String& text) {
    auto* self=static_cast<Frame*>(opaque);if(!self||!slot)return false;
    self->writeString(*slot,text.substring(0,16383).toRawUTF8());return true;
}
static bool readUtf8FromStringArgument(void* opaque,Reference slot,juce::String& text) {
    auto* self=static_cast<Frame*>(opaque);if(!self||!slot)return false;
    auto str=self->string(*slot);text=juce::String::fromUTF8(str.c_str(),(int)str.size());return !text.isEmpty();
}
void syncMemRange(const double* source,int64_t base,int count) {
    auto& shared=scriptState();
    if(!shared.mem||base<0||count<0||base+count>shared.memN){memoryFault=true;return;}
    for(int i=0;i<count;++i)shared.mem[base+i]=source[i];
}
void ensureMemSize(int64_t end) { if(end>scriptState().memN)memoryFault=true; }
void markFileMemRange(int64_t,int) {} // Shared cells are already visible to DSP.
void resetDrawing() {
    commands.clear();pendingImageBytes=0;framebufferDirty=false;frameRecording=false;
    imageBank=GfxImageBank();imageWidths.fill(0);imageHeights.fill(0);imageResourceAttempted.fill(false);
    fonts={};fontCache.clear();currentFont=0;fileHandles.clear();freeFileHandles.clear();
    pendingDroppedFileBatches.clear();activeDroppedFiles.clear();
}
  FileHandle* getFileHandle(double value) noexcept
  {
    if (!std::isfinite((double)value)) return nullptr;
    const int64_t hid = jsfx_gfx::jsfxTruncIndexLikeAot((double)value);
    if (hid <= 0) return nullptr;
    const int64_t idx64 = hid - 1;
    if (idx64 < 0 || idx64 >= (int64_t)fileHandles.size()) return nullptr;
    auto& h = fileHandles[(size_t)idx64];
    return h.files.empty() ? nullptr : &h;
  }

  static bool parseFileNumberToken(juce::String token, double& out)
  {
    token = token.trim();
    if (token.isEmpty()) return false;

    auto parseInteger = [] (const juce::String& text, int base, int64_t& value) -> bool
    {
      const auto utf8 = text.toRawUTF8();
      if (utf8 == nullptr || *utf8 == 0) return false;
      char* end = nullptr;
      const long long v = std::strtoll(utf8, &end, base);
      if (end == utf8 || (end != nullptr && *end != 0)) return false;
      value = (int64_t)v;
      return true;
    };

    int64_t iv = 0;
    if (token.startsWithIgnoreCase("0x") && parseInteger(token.substring(2), 16, iv))
    { out = (double)iv; return true; }
    if (token.startsWithChar('x') && parseInteger(token.substring(1), 16, iv))
    { out = (double)iv; return true; }
    if (token.startsWithChar('b') && parseInteger(token.substring(1), 2, iv))
    { out = (double)iv; return true; }

    const auto utf8 = token.toRawUTF8();
    char* end = nullptr;
    const double v = std::strtod(utf8, &end);
    if (end == utf8 || (end != nullptr && *end != 0) || !std::isfinite(v)) return false;
    out = v;
    return true;
  }

  std::unique_ptr<FileData> loadFileData(const juce::File& file)
  {
    if (!file.existsAsFile()) return nullptr;

    if (auto reader = std::unique_ptr<juce::AudioFormatReader>(fileFormatManager.createReaderFor(file)))
    {
      auto out = std::make_unique<FileData>();
      out->textMode = false;
      out->channels = std::max(1, std::min(64, (int)reader->numChannels));
      out->frames = std::max<int64_t>(0, reader->lengthInSamples);
      out->sampleRate = reader->sampleRate;

      if (out->frames <= 0) return out;
      if (out->frames > std::numeric_limits<int64_t>::max() / out->channels) return nullptr;
      const int64_t totalItems = out->frames * (int64_t)out->channels;

      // GFX file loading is intentionally eager so file_avail()/file_mem() have
      // REAPER-like immediate semantics.  Keep a hard safety ceiling against a
      // corrupt header requesting an absurd allocation.
      constexpr int64_t kMaxDecodedItems = 64ll * 1024ll * 1024ll; // 512 MiB of doubles
      if (totalItems <= 0 || totalItems > kMaxDecodedItems) return nullptr;

      try { out->items.resize((size_t)totalItems); }
      catch (...) { return nullptr; }

      juce::AudioBuffer<float> temp;
      constexpr int kChunk = 65536;
      int64_t decoded = 0;
      for (int64_t pos = 0; pos < out->frames; pos += kChunk)
      {
        const int n = (int)std::min<int64_t>(kChunk, out->frames - pos);
        temp.setSize(out->channels, n, false, false, true);
        temp.clear();
        if (!reader->read(&temp, 0, n, pos, true, true)) break;

        for (int i = 0; i < n; ++i)
        {
          const size_t base = (size_t)((pos + i) * out->channels);
          for (int ch = 0; ch < out->channels; ++ch)
            out->items[base + (size_t)ch] = (double)temp.getSample(ch, i);
        }
        decoded = pos + n;
      }

      if (decoded < out->frames)
      {
        out->frames = std::max<int64_t>(0, decoded);
        out->items.resize((size_t)(out->frames * out->channels));
      }
      return out;
    }

    const auto content = file.loadFileAsString();
    if (content.isEmpty() && file.getSize() > 0) return nullptr;

    auto out = std::make_unique<FileData>();
    out->textMode = file.getFileExtension().equalsIgnoreCase(".txt");

    // file_string() is line-oriented. Preserve newline characters except at
    // EOF, matching the normal-file behaviour described by REAPER.
    int pos = 0;
    while (pos < content.length())
    {
      int nl = content.indexOfChar(pos, '\n');
      if (nl < 0)
      {
        out->lines.push_back(content.substring(pos));
        break;
      }
      out->lines.push_back(content.substring(pos, nl + 1));
      pos = nl + 1;
    }
    if (content.isEmpty()) out->lines.clear();

    juce::StringArray lines;
    lines.addLines(content);
    for (auto line : lines)
    {
      const int semi = line.indexOfChar(';');
      const int hash = line.indexOfChar('#');
      int cut = -1;
      if (semi >= 0) cut = semi;
      if (hash >= 0) cut = cut < 0 ? hash : std::min(cut, hash);
      if (cut >= 0) line = line.substring(0, cut);

      // Native text records are newline/comma delimited. Whitespace within a
      // record is also accepted for the common numeric-data case. Symbolic
      // expression records remain a compatibility TODO; plain/binary/hex data
      // (the overwhelmingly common JSFX resource form) is handled here.
      line = line.replaceCharacter(',', ' ').trim();
      if (line.isEmpty()) continue;
      juce::StringArray toks;
      toks.addTokens(line, " \t", "");
      toks.removeEmptyStrings();
      for (auto tok : toks)
      {
        double v = 0.0;
        if (parseFileNumberToken(tok, v)) out->items.push_back(v);
      }
    }
    return out;
  }

  bool ensureFileData(FileHandle& h)
  {
    if (h.selectedFile < 0 || h.selectedFile >= (int)h.files.size()) return false;
    if (h.data) return true;
    h.data = loadFileData(h.files[(size_t)h.selectedFile]);
    return h.data != nullptr;
  }

  bool resampleFileData(FileHandle& h, double targetRate)
  {
    if (!ensureFileData(h) || !h.data || h.data->channels <= 0 || h.data->frames <= 0)
      return false;
    if (!std::isfinite(targetRate) || targetRate < 1000.0 || targetRate > 768000.0)
      return false;
    if (!(h.data->sampleRate > 0.0) || std::abs(h.data->sampleRate - targetRate) < 1.0e-9)
    { h.data->sampleRate = targetRate; return true; }
    if (h.readMode != FileData::ReadMode::none || h.cursor != 0)
      return false; // REAPER requires rqsr before file_var()/file_mem().

    const int chs = h.data->channels;
    const int64_t srcFrames = h.data->frames;
    const double ratio = targetRate / h.data->sampleRate;
    const int64_t dstFrames = std::max<int64_t>(1, (int64_t)std::llround(srcFrames * ratio));
    constexpr int64_t kMaxDecodedItems = 64ll * 1024ll * 1024ll;
    if (dstFrames > kMaxDecodedItems / chs) return false;

    std::vector<double> dst;
    try { dst.resize((size_t)(dstFrames * chs)); }
    catch (...) { return false; }

    const double sourceStep = h.data->sampleRate / targetRate;
    for (int64_t df = 0; df < dstFrames; ++df)
    {
      const double sp = std::min<double>((double)(srcFrames - 1), (double)df * sourceStep);
      const int64_t i0 = (int64_t)std::floor(sp);
      const int64_t i1 = std::min<int64_t>(srcFrames - 1, i0 + 1);
      const double frac = sp - (double)i0;
      for (int ch = 0; ch < chs; ++ch)
      {
        const double a = h.data->items[(size_t)(i0 * chs + ch)];
        const double b = h.data->items[(size_t)(i1 * chs + ch)];
        dst[(size_t)(df * chs + ch)] = a + (b - a) * frac;
      }
    }

    h.data->items.swap(dst);
    h.data->frames = dstFrames;
    h.data->sampleRate = targetRate;
    return true;
  }

  std::vector<juce::File> resolveFileOpenTarget(Reference arg)
  {
    std::vector<juce::File> files;
    if (!arg) return files;

    const double raw = (double)*arg;
    const int64_t numeric = jsfx_gfx::jsfxTruncIndexLikeAot(raw);

    // Declared filename:N slots win over user-string slots of the same small
    // integer.  Otherwise a populated EEL string handle is a path, which is
    // what gfx_getdropfile(0, 14); file_open(14); relies on.
    if (numeric >= 0 && numeric < (int64_t)fileResourceDeclared.size()
        && std::abs(raw - (double)numeric) < 1.0e-8
        && fileResourceDeclared[(size_t)numeric])
    {
      if (fileResolver)
        return fileResolver((int)numeric, fileResourceNames[(size_t)numeric]);
      if (fileResourceNames[(size_t)numeric].isNotEmpty())
        files.emplace_back(fileResourceNames[(size_t)numeric]);
      return files;
    }

    juce::String path;
    if (readUtf8FromStringArgument(this, arg, path) && path.isNotEmpty())
    {
      if (fileResolver) return fileResolver(-1, path);
      files.emplace_back(path);
    }
    return files;
  }

#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
  uint64_t testFileOpens=0,testFileValues=0;
#endif
  double openFileCommon(Reference arg)
  {
    auto files = resolveFileOpenTarget(arg);
    files.erase(std::remove_if(files.begin(), files.end(), [] (const juce::File& f) { return !f.existsAsFile(); }), files.end());
    if (files.empty()) return -1.0;

    int idx = -1;
    if (!freeFileHandles.empty())
    {
      idx = freeFileHandles.back();
      freeFileHandles.pop_back();
    }
    else
    {
      idx = (int)fileHandles.size();
      fileHandles.emplace_back();
    }

    auto& h = fileHandles[(size_t)idx];
    h = FileHandle{};
    h.files = std::move(files);
    if (!ensureFileData(h))
    {
      h = FileHandle{};
      freeFileHandles.push_back(idx);
      return -1.0;
    }
#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
    ++testFileOpens;
#endif
    return (double)(idx + 1);
  }

  static double eel_file_open(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    return self && np >= 1 && parms && parms[0] ? self->openFileCommon(parms[0]) : -1.0;
  }

  static double eel_file_open_multi(void* opaque, intptr_t np, Reference* parms)
  {
    // DSP-JSFX extension: the resolver may return multiple enhanced-slot paths.
    return eel_file_open(opaque, np, parms);
  }

  static double eel_file_close(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    if (!self || np < 1 || !parms || !parms[0]) return 0.0;
    auto* h = self->getFileHandle(*parms[0]);
    if (!h) return 0.0;
    const int idx = (int)(h - self->fileHandles.data());
    self->fileHandles[(size_t)idx] = FileHandle{};
    self->freeFileHandles.push_back(idx);
    return 0.0;
  }

  static double eel_file_rewind(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 1 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h) return 0.0;
    h->cursor = 0;
    h->lineCursor = 0;
    h->readMode = FileData::ReadMode::none;
    return 0.0;
  }

  static double eel_file_seek(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 2 && parms && parms[0] && parms[1] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h)) return 0.0;
    int64_t off = jsfx_gfx::jsfxTruncIndexLikeAot((double)*parms[1]);
    if (off < 0) off = 0;
    const int64_t maxItems = h->data ? (int64_t)h->data->items.size() : 0;
    h->cursor = std::min(off, maxItems);
    h->lineCursor = std::min<int64_t>(off, h->data ? (int64_t)h->data->lines.size() : 0);
    return (double)h->cursor;
  }

  static double eel_file_avail(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 1 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h) || !h->data) return 0.0;

    if (h->readMode == FileData::ReadMode::string)
      return h->lineCursor < (int64_t)h->data->lines.size() ? 1.0 : 0.0;

    const int64_t remaining = (int64_t)h->data->items.size() - h->cursor;
    if (h->data->textMode) return remaining > 0 ? 1.0 : 0.0;
    return remaining > 0 ? (double)remaining : 0.0;
  }

  static double eel_file_text(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 1 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    return h && self->ensureFileData(*h) && h->data && h->data->textMode ? 1.0 : 0.0;
  }

  static double eel_file_riff(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 3 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h) || !h->data || h->data->channels <= 0)
    {
      if (np >= 2 && parms[1]) *parms[1] = 0.0;
      if (np >= 3 && parms[2]) *parms[2] = 0.0;
      return 0.0;
    }

    const uint32_t request = (np >= 2 && parms[1] && std::isfinite((double)*parms[1]))
                           ? (uint32_t)std::llround((double)*parms[1]) : 0u;
    if (request == kRqsrFourCC && np >= 3 && parms[2])
      (void)self->resampleFileData(*h, (double)*parms[2]);

    if (parms[1]) *parms[1] = (double)h->data->channels;
    if (parms[2]) *parms[2] = (double)h->data->sampleRate;
    return 1.0;
  }

  static double eel_file_var(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 2 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h) || !h->data || h->cursor < 0
        || h->cursor >= (int64_t)h->data->items.size())
    {
      if (np >= 2 && parms[1]) *parms[1] = 0.0;
      return 0.0;
    }

    const double v = h->data->items[(size_t)h->cursor++];
    h->readMode = FileData::ReadMode::numeric;
#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
    ++self->testFileValues;
#endif
    if (parms[1]) *parms[1] = (double)v;
    return (double)v;
  }

  static double eel_file_mem(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 3 && parms && parms[0] && parms[1] && parms[2]
            ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h) || !h->data) return 0.0;

    int64_t dst = jsfx_gfx::jsfxTruncIndexLikeAot((double)*parms[1]);
    int64_t len = jsfx_gfx::jsfxTruncIndexLikeAot((double)*parms[2]);
    if (dst < 0) dst = 0;
    if (len <= 0) return 0.0;
    const int64_t available = (int64_t)h->data->items.size() - h->cursor;
    if (available <= 0) return 0.0;
    int64_t n64 = std::min(len, available);

    const int64_t eelMax = self->scriptState().memN;
    if (dst >= eelMax) return 0.0;
    n64 = std::min(n64, eelMax - dst);
    n64 = std::min<int64_t>(n64, std::numeric_limits<int>::max());
    if (n64 <= 0) return 0.0;

    const int n = (int)n64;
    self->ensureMemSize(dst + n);
    self->syncMemRange(h->data->items.data() + h->cursor, dst, n);
    self->markFileMemRange(dst, n);
    h->cursor += n;
    h->readMode = FileData::ReadMode::numeric;
#if ZA_NATIVE_GFX_TEST_RUNNER || ZA_SAMPLE_GFX_TEST_RUNNER
    self->testFileValues+=(uint64_t)n;
#endif
    return (double)n;
  }

  static double eel_file_string(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 2 && parms && parms[0] && parms[1]
            ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h || !self->ensureFileData(*h) || !h->data
        || h->lineCursor < 0 || h->lineCursor >= (int64_t)h->data->lines.size())
    {
      if (self && np >= 2 && parms && parms[1]) writeUtf8ToStringArgument(self, parms[1], {});
      return 0.0;
    }

    const auto line = h->data->lines[(size_t)h->lineCursor++];
    h->readMode = FileData::ReadMode::string;
    return writeUtf8ToStringArgument(self, parms[1], line) ? 1.0 : 0.0;
  }

  static double eel_file_multi_count(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 1 && parms && parms[0] ? self->getFileHandle(*parms[0]) : nullptr;
    return h ? (double)h->files.size() : 0.0;
  }

  static double eel_file_multi_select(void* opaque, intptr_t np, Reference* parms)
  {
    auto* self = (Frame*)opaque;
    auto* h = self && np >= 2 && parms && parms[0] && parms[1] ? self->getFileHandle(*parms[0]) : nullptr;
    if (!h) return 0.0;
    const int64_t idx = jsfx_gfx::jsfxTruncIndexLikeAot((double)*parms[1]);
    if (idx < 0 || idx >= (int64_t)h->files.size()) return 0.0;
    h->selectedFile = (int)idx;
    h->data.reset();
    h->cursor = 0;
    h->lineCursor = 0;
    h->readMode = FileData::ReadMode::none;
    return self->ensureFileData(*h) ? 1.0 : 0.0;
  }
