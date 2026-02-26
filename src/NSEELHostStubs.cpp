#include <mutex>
#include "WDL/eel2/ns-eel.h"

static std::mutex g_eelGlobalMutex;

extern "C" void NSEEL_HOSTSTUB_EnterMutex(void)
{
  g_eelGlobalMutex.lock();
}

extern "C" void NSEEL_HOSTSTUB_LeaveMutex(void)
{
  g_eelGlobalMutex.unlock();
}