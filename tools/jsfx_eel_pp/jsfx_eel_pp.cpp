#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "WDL/wdlstring.h"
#include "WDL/ptrlist.h"
#include "WDL/eel2/eel_pproc.h"

void NSEEL_HOSTSTUB_EnterMutex() {}
void NSEEL_HOSTSTUB_LeaveMutex() {}

namespace
{
struct Definition
{
    std::string name;
    double value = 0.0;
};

[[noreturn]] void usage(const char* exe, int code)
{
    std::fprintf(stderr,
                 "Usage: %s [--include DIR] [--define NAME=VALUE] [--max-size BYTES]\n"
                 "Reads UTF-8 JSFX text from stdin and writes preprocessed text to stdout.\n",
                 exe ? exe : "jsfx_eel_pp");
    std::exit(code);
}

bool parseDouble(const char* text, double& out)
{
    if (!text || !*text)
        return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(text, &end);
    if (errno == ERANGE || end == text || (end && *end != '\0') || !std::isfinite(v))
        return false;
    out = v;
    return true;
}

bool validName(const std::string& name)
{
    if (name.empty())
        return false;
    const auto first = static_cast<unsigned char>(name.front());
    if (!(name.front() == '_' || (first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z')))
        return false;
    for (char c : name)
    {
        const auto u = static_cast<unsigned char>(c);
        if (!(c == '_' || c == '.' || (u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') ||
              (u >= '0' && u <= '9')))
            return false;
    }
    return true;
}

std::string readStdin()
{
    std::string data;
    char buf[16384];
    while (true)
    {
        const size_t n = std::fread(buf, 1, sizeof(buf), stdin);
        if (n)
            data.append(buf, n);
        if (n < sizeof(buf))
        {
            if (std::ferror(stdin))
            {
                std::fprintf(stderr, "Error: failed reading stdin\n");
                std::exit(2);
            }
            break;
        }
    }
    return data;
}
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> includePaths;
    std::vector<Definition> definitions;
    int maxSize = 64 << 20;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "-h" || arg == "--help")
            usage(argv[0], 0);
        if (arg == "--include")
        {
            if (++i >= argc)
                usage(argv[0], 2);
            includePaths.emplace_back(argv[i]);
            continue;
        }
        if (arg == "--define")
        {
            if (++i >= argc)
                usage(argv[0], 2);
            const std::string spec = argv[i] ? argv[i] : "";
            const size_t eq = spec.find('=');
            if (eq == std::string::npos)
            {
                std::fprintf(stderr, "Error: invalid --define %s (expected NAME=VALUE)\n", spec.c_str());
                return 2;
            }
            Definition d;
            d.name = spec.substr(0, eq);
            if (!validName(d.name) || !parseDouble(spec.c_str() + eq + 1, d.value))
            {
                std::fprintf(stderr, "Error: invalid --define %s\n", spec.c_str());
                return 2;
            }
            definitions.push_back(std::move(d));
            continue;
        }
        if (arg == "--max-size")
        {
            if (++i >= argc)
                usage(argv[0], 2);
            char* end = nullptr;
            const long long v = std::strtoll(argv[i], &end, 10);
            if (!end || *end != '\0' || v < 1024 || v > (1024LL << 20))
            {
                std::fprintf(stderr, "Error: invalid --max-size %s\n", argv[i]);
                return 2;
            }
            maxSize = static_cast<int>(v);
            continue;
        }

        std::fprintf(stderr, "Error: unknown argument %s\n", arg.c_str());
        usage(argv[0], 2);
    }

    const std::string input = readStdin();
    WDL_FastString output;
    EEL2_PreProcessor pproc(maxSize);

    // eel_pproc.h searches include paths from newest to oldest.  The Python
    // caller deliberately passes broad roots first and the importing file's
    // directory last, so local includes win just as they do in REAPER.
    for (const std::string& path : includePaths)
        pproc.m_include_paths.Add(path.c_str());
    for (const Definition& d : definitions)
        pproc.define(d.name.c_str(), d.value);

    const char* error = pproc.preprocess(input.c_str(), &output);
    if (error)
    {
        std::fprintf(stderr, "Error: %s\n", error);
        return 3;
    }

    if (output.GetLength() > 0)
    {
        const size_t written = std::fwrite(output.Get(), 1, static_cast<size_t>(output.GetLength()), stdout);
        if (written != static_cast<size_t>(output.GetLength()))
        {
            std::fprintf(stderr, "Error: failed writing stdout\n");
            return 4;
        }
    }
    return 0;
}
