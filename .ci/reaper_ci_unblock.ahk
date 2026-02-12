#Requires AutoHotkey v2.0
#SingleInstance Force
SetTitleMatchMode 2
DetectHiddenWindows false

; Write a heartbeat so you know the script actually ran
DirCreate(A_ScriptDir "\out")
FileAppend("ahk started: " A_Now "`n", A_ScriptDir "\out\ahk_unblock.log")

deadline := A_TickCount + 20*60*1000

Loop {
    if (A_TickCount > deadline) {
        FileAppend("ahk exiting (deadline)`n", A_ScriptDir "\out\ahk_unblock.log")
        ExitApp 0
    }

    ; Enumerate REAPER windows and target ONLY the modal whose title is EXACTLY "REAPER"
    for hwnd in WinGetList("ahk_exe reaper.exe") {
        title := WinGetTitle("ahk_id " hwnd)

        if (title = "REAPER") {
            ; This is the modal dialog in your screenshot.
            WinActivate("ahk_id " hwnd)
            Sleep 80

            ; Prefer clicking the "No" button directly (typical MessageBox layout: Button2)
            ok := false
            try {
                ControlClick("Button2", "ahk_id " hwnd) ; "No"
                ok := true
            } catch {
                ok := false
            }

            if (!ok) {
                ; Fallback: Alt+N (accelerator for "No")
                Send "!n"
            }

            FileAppend("dismissed modal: " A_Now " hwnd=" hwnd "`n", A_ScriptDir "\out\ahk_unblock.log")
            Sleep 250
        }
    }

    Sleep 150
}
