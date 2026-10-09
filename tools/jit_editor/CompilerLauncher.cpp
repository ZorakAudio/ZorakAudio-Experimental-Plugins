// Linux process supervision, separate from compiler/runtime semantics.
// The launcher owns a process group containing Python and any Faust child.
// Parent death or cancellation kills that group, never the DAW's process group.
#include <cerrno>
#include <cstdlib>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

static void parentDied(int) { kill(-getpid(), SIGKILL); }
int main(int argc, char** argv) {
    if (argc < 4 || getpgrp() != getpid()) return 125;
    const auto parent = static_cast<pid_t>(std::strtol(argv[1], nullptr, 10));
    struct sigaction action {};
    action.sa_handler = parentDied;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, nullptr) || prctl(PR_SET_PDEATHSIG, SIGTERM)) return 125;
    if (getppid() != parent) parentDied(0);
    const auto child = fork();
    if (child < 0) return 125;
    if (child == 0) { execv(argv[2], argv + 2); _exit(127); }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) if (errno != EINTR) return 125;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
