#pragma once
#include <juce_core/juce_core.h>
namespace jsfx_gfx {
static constexpr int SHOWMENU_NB_NONE_VALUE     = 0;
static constexpr int SHOWMENU_NB_PENDING_VALUE  = -1;
static constexpr int SHOWMENU_NB_CANCELED_VALUE = -2;

struct AsyncMenuPort
{
  virtual ~AsyncMenuPort() = default;

  // Worker-thread modal menu call. This blocks the dedicated @gfx worker until
  // the UI-side menu is dismissed, while keeping the message thread responsive.
  // That preserves classic JSFX gfx_showmenu() semantics.
  virtual int showMenuModal(const juce::String& description, int x, int y) = 0;

  // Explicit non-blocking menu API.
  //
  // open() returns 1 on success, 0 if no menu was opened.
  // poll() returns one of:
  //   SHOWMENU_NB_NONE_VALUE      (0)  -> no active async menu / no pending result
  //   SHOWMENU_NB_PENDING_VALUE   (-1) -> async menu still open or waiting
  //   SHOWMENU_NB_CANCELED_VALUE  (-2) -> async menu canceled / clicked away
  //   > 0                               -> selected 1-based menu index
  // cancel() returns 1 if a pending/open async menu was canceled, 0 otherwise.
  virtual int showMenuNonBlockingOpen(const juce::String& description, int x, int y) = 0;
  virtual int showMenuNonBlockingPoll() = 0;
  virtual int showMenuNonBlockingCancel() = 0;
  virtual void cancelAll() = 0;
};


}
