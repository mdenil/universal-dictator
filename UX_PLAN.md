# Universal Dictator: UX changes, round 2

This plan covers seven pieces of feedback on the app as it runs today (commit `fc403d3`), revised after your first review. Each section says what changes, how it will be built and how it will be tested.

The designs are in Figma: [Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl).
- **"Settings (draft)"**: the right-click menu, the Settings window, and the Settings window while a dictation is paused.
- **"Menu-bar icon"** (in Components): the colour states. The Paused panel state uses the yellow icon.

## 1. A paused dictation survives closing the panel

**Today:** closing the panel always ends the dictation, and the next click starts a new one.

**New rule:** closing the panel while **paused** keeps the dictation. The next click on the icon opens the panel still paused, with the text in place. Resume carries on from where you left off. In every other state, closing still ends the dictation as it does now:

| State when the panel closes | What happens | Next click on the icon |
|---|---|---|
| Listening | Dictation ends; text goes to History | New dictation, listening |
| **Paused** | **Kept: microphone already off, text (with your edits) and timer kept** | **Same dictation, paused; Resume continues it** |
| Stopped (editing) | Dictation ends; text, with your edits, goes to History | New dictation, listening |

Details:
- Stop, ⌘C and Copy ⌘C all work on a reopened paused dictation, the same as before closing.
- Quitting the app with a paused dictation saves its text to History, so nothing is lost.
- The model can't be changed while a dictation is paused, since the paused dictation belongs to the loaded model. The Model menu in Settings is disabled, and its tooltip says why.

**Build:**
- `DictationController.panelClosed()` leaves the `.paused` case alone.
- `panelOpened()` does nothing for `.paused`.
- A new `finishForQuit()`, called from `applicationWillTerminate`, archives the text.

**Tests:**
- pause, close, reopen → still paused, same text, the microphone still off; Resume → listening, and the same session gets the audio;
- pause, close, reopen, Stop → done with the full text;
- listening, close → ends, as today;
- quit while paused → the text is in History.

## 2. The menu-bar icon speaks in colour

**Decision (yours):** one shape, the waveform, in every state, with the colour showing the state:

| State | Icon |
|---|---|
| Idle, and while the model gets ready | as now: white on a dark menu bar, black on a light one |
| Listening | red |
| Paused (panel open or closed) | yellow |

The listening badge and the pause-circle icon go away.

One exception I'd keep, unless you say otherwise: **No microphone access** and **Couldn't load the model** keep the warning-triangle shape. Those states need you to do something, and a colour alone wouldn't say what is wrong.

**Build:**
- In `MenuBarPanel.updateIcon()`, Listening and Paused use the `waveform` symbol drawn in the system red or yellow. They are non-template images, so macOS doesn't recolour them; the system colours adapt to light and dark menu bars.
- Idle stays a template image, as now.

**Test:** by hand in both a light and a dark menu bar, plus the existing phase log.

## 3. The menu-bar icon disappearing while dictating: confirm only

**Likely cause:** your MacBook has a notch, and our icon sits immediately right of it: the log puts the icon at about x 991, and the notch ends at x 948. When the microphone turns on, macOS adds its orange microphone indicator to the right end of the menu bar. That pushes every status icon left by one slot, and ours, the leftmost, slides behind the notch.

**Decision (yours):** if this is confirmed, no fix.

**Build:** a short run of `--diagnose-start` that logs whether the icon's window is visible before and after the microphone starts. I'll report the result. If it turns out to be something else, I'll come back to you.

## 4. The text box: no highlight while dictating; editable while paused

**Today:** the live text is selectable in every state, and the same text view stays on screen through listening, paused and listening again. A selection you make while paused therefore stays when you resume, and new words flow in under the highlight.

**New rule:**

| State | Text box |
|---|---|
| Listening | Shows the live text; it can't be selected or edited, so no highlight ever shows |
| **Paused** | **Editable**, like after Stop |
| Stopped | Editable, as today |

**Resume after editing:** your edited text stays exactly as you left it, and new words are appended after it. Stop, ⌘C and Copy ⌘C give your edited text plus everything said since.

**Size: small, as you expected.** The engine already keeps a dictation as a list of segments, one per stretch between pauses. Resume swaps your edited text in as the earlier segments, and the rest is unchanged. That's about 40 lines across `DictationSession` and the controller, plus tests. Also:
- the session's `pause()` waits until the last words are finished, so the editor opens with the complete text;
- a selection can't survive into Listening, because the editor is replaced by the read-only live view.

**Tests:**
- pause, edit, resume, speak, Stop → your edit plus the new words;
- pause, edit, close, reopen → the edit is still there;
- pause, delete everything, resume → only the new words;
- pause without editing, resume → same as today.

## 5. "Copy so far" becomes **Copy ⌘C**, doing exactly what ⌘C does

**Decision (yours):** the button reads **Copy ⌘C**: the word Copy, followed by the shortcut in lighter text. Clicking it does exactly what ⌘C does: the microphone stops, the last words are finished, the whole text is copied, it goes to History, and the panel closes. Both run the same code path.

The "copy without stopping" action goes away. The other buttons are unchanged: Pause/Resume (Space) and Stop (Return).

**Build:** remove `copySoFar()` and `copiedSoFar` from the controller. The button and the ⌘C shortcut both call one `copyAndClose` closure in `PanelView`.

**Test:** the existing `testStopAndCopy`, plus a snapshot of the new footer.

## 6. New "Getting ready" text

Exactly as you wrote it:

```
Getting things ready for you…
This usually takes less than a minute and only happens the first time you open the app.
```

It is applied in `PanelView.swift` and in the Figma "Preparing" state.

## 7. History is kept between launches

**Decision (yours):** history persists between runs. It keeps 50 entries by default, and the limit is set in Settings.

**Build:**
- The entries are saved as a small JSON file in `~/Library/Application Support/dev.universaldictator.UniversalDictator/history.json`. It stays on this Mac, written after every change and read at launch.
- The controller gets a `HistoryStore` protocol: a file store in the app, an in-memory one in tests.
- Lowering the limit drops the oldest entries straight away.
- The History window (right-click → Show All History…) is built as drafted in "History (draft)": search, and Copy, **Edit** and Delete per entry. Each entry's model is shown as Fast or Accurate.
- **Editing a history entry:** Edit turns the entry's text into an editable box.
  - **Done** (⌘Return) saves the change to the history file, and the entry is marked "edited".
  - **Cancel** (Esc) drops the change.
  - Done is disabled while the text is empty; Delete is how an entry is removed.
  - Copy, from the window or the right-click menu, copies the edited text.
  - The entry keeps its place and time.

**Tests:**
- save, then load in a new controller → same entries, same order;
- a limit of 3 with 5 dictations → the 3 newest are kept;
- lowering the limit trims the list;
- Clear History empties the list and the file;
- a damaged file → empty history, without crashing;
- editing an entry → the new text is saved, survives a reload, keeps its place, and is marked edited;
- cancelling an edit → no change.

## 8. A Settings window

It opens from **Settings…** in the right-click menu, and with **⌘,** while the panel is open.

Contents, drafted in Figma "Settings (draft)", with no descriptive text:

Labels are left-aligned and controls right-aligned, so the controls line up on the right edge:

| Label (left) | Controls (right) |
|---|---|
| Model | a menu: **Fast** (Nemotron 160 ms, the default) or **Accurate** (Nemotron 560 ms) |
| History | **Clear History…** (asks for confirmation, then deletes everything), then a menu: 10, 25, **50**, 100 or 200 |

The model choice lives only in Settings. The panel header loses its model menu and keeps the icon and title.

The right-click menu, top to bottom:
- the five recent dictations;
- "Show All History…";
- "Settings… ⌘,";
- "Quit".

**Switching models stays fast.** Two caches already keep a model ready once it has been prepared:
- **Disk:** macOS keeps each model's Neural Engine compile on disk, so a model prepared once isn't prepared again on later launches.
- **Memory:** once loaded, a model stays in memory for the rest of the run, so switching back to it is immediate.

Only the first switch to a model never used on this Mac shows "Getting ready".

I'll measure three things and report them:
- the time to switch to a model, back and forth, in one run;
- the time to switch after a relaunch;
- the extra memory used by keeping both models loaded.

If the memory cost turns out to be large, I'll come back to you before choosing between memory and switch time.

**Build:**
- the SwiftUI `Settings` scene, which is already in the app but empty;
- a `SettingsView` bound to the controller;
- `ModelChoice.displayName` becomes "Fast" or "Accurate";
- the right-click menu as listed above.

**Tests:**
- changing the history limit trims the history; the history tests above cover this;
- changing the model while paused is refused;
- switching away and back doesn't reload the first model;
- snapshots of the Settings window.

## Order of work

1. Getting-ready text, the Copy ⌘C button and the icon colours.
2. Paused dictations survive closing; editable while paused; no highlight while listening.
3. Confirm the icon-disappearing cause and report.
4. Persistent history, then Settings with Fast and Accurate, then the History menu and window.

Each step is committed separately with its tests passing, and the app is rebuilt and restarted after each, so you can try it as it lands.
