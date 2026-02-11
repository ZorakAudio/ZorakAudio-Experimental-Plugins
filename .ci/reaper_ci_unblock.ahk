#Requires AutoHotkey v2.0
#SingleInstance Force
SetTitleMatchMode 2
DetectHiddenWindows false

; Wait up to 5 minutes
deadline := A_TickCount + 5*60*1000

Loop {
    if (A_TickCount > deadline)
        ExitApp 0

    ; The blocking dialog is a separate window titled "REAPER" (modal).
    ; We'll activate it and send Alt+N (the No button).
    if WinExist("REAPER") {
        hwnd := WinExist("REAPER")
        ; Only act if it's a dialog-style window (often class #32770 for dialogs).
        cls := WinGetClass("ahk_id " hwnd)
        if (cls = "#32770") {
            WinActivate("ahk_id " hwnd)
            Sleep 80
            Send "!n"        ; Alt+N => clicks "No" on that dialog
            Sleep 200
        }
    }

    ; Optional: evaluation nag windows sometimes are dialogs too; Enter is ok ONLY on dialogs.
    ; This avoids affecting the main REAPER window.
    ; If you see a different modal later, this dismisses default buttons safely.
    for w in WinGetList("ahk_exe reaper.exe") {
        cls2 := WinGetClass("ahk_id " w)
        if (cls2 = "#32770") {
            WinActivate("ahk_id " w)
            Sleep 80
            Send "{Enter}"
            Sleep 200
        }
    }

    Sleep 150
}
