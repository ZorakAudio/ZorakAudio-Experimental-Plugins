#Requires AutoHotkey v2.0
#SingleInstance Force
SetTitleMatchMode 2
DetectHiddenWindows false

; Keep trying for up to 15 minutes (your CI timeout window)
deadline := A_TickCount + 15*60*1000

Loop {
    if (A_TickCount > deadline)
        ExitApp 0

    ; --- Audio device prompt (modal) ---
    ; Window title is usually "REAPER" for the dialog.
    ; We avoid coordinates: click the "No" button by control name if possible, else tab+enter.
    if WinExist("REAPER") {
        ; Try to detect the specific prompt text in the dialog
        t := WinGetText("REAPER")
        if InStr(t, "You have not yet selected an audio device") {
            WinActivate("REAPER")
            Sleep 100

            ; Attempt direct ControlClick on the "No" button.
            ; On most systems the buttons are Button1/Button2, with "No" often Button2.
            try ControlClick("Button2", "REAPER")
            catch {
                ; Fallback: Tab to "No" then Enter
                ; (Usually focus starts on Yes, so one tab goes to No)
                Send "{Tab}{Enter}"
            }
            Sleep 250
        }
    }

    ; --- Optional: evaluation nag dialog(s) ---
    ; Some builds show a modal “Still evaluating” / “Purchase” dialog.
    ; We just press Enter on any small modal window owned by REAPER.
    ; (This is intentionally generic but low-risk.)
    hwnd := WinExist("ahk_exe reaper.exe")
    if (hwnd) {
        ; If there is any active dialog-style window, Enter often dismisses default button.
        ; We only do this when REAPER is active to avoid messing with other apps.
        if WinActive("ahk_exe reaper.exe") {
            Send "{Enter}"
        }
    }

    Sleep 200
}
