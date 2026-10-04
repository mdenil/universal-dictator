# Universal Dictator: direct dictation

Direct dictation adds a global shortcut that dictates straight into the text field in focus, in any app, without opening the panel. Everything the app does today stays exactly as it is: the panel, pause and resume, Copy ⌘C, History and Settings. Direct dictation is an extra way in.

The new UI is drafted in Figma under "Direct dictation (draft)": [Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl). It has three parts: the Settings window with its new rows, a panel state for the missing permission, and an overview of direct dictation in another app.

## What it does

1. **Double-tap Left Shift** (the default shortcut) while typing in any app. The microphone starts; when it is live, a moment later, the menu-bar icon turns red and a short sound plays ("Tink"). Speak then. No window opens.
2. **Speak.** Each word is typed into the focused text field as if you had typed it, as soon as it is final. That is about 0.3 s after you start the next word. The word still being recognised is never typed, so nothing ever needs to be erased and retyped.
3. **Stop**, in one of two modes chosen in Settings:
   - **Toggle** (default): double-tap to start, a single tap of Shift to stop. A key-combination shortcut stops with a second press.
   - **Hold to talk**: **Shift, Shift-hold (talk), release.** Tap Shift once, press it again and keep holding while you speak; releasing it stops.

   Stopping types the last word, turns the microphone off and the icon back to normal.
4. The dictation goes to History, marked **typed**, so the text is never lost even if it landed in the wrong place.

The text goes wherever keyboard focus is as each word arrives. Clicking into another field mid-dictation sends the rest there, just as typing would.

## How typing works

The app sends each word as keyboard events carrying the text (`CGEvent` with a Unicode string). To the receiving app it looks like fast typing, so it works in native apps, browsers, Electron apps such as Slack, and terminals. Undo in the receiving app removes the typed text the way it removes typing.

- **Spacing:**
  - Words are typed with the spacing the model gives them.
  - The first word of a dictation gets no leading space, so the text starts exactly at the cursor.
  - Each later word starts with a space.
- **Accessibility permission:** both typing into other apps and noticing a double tap of Shift in other apps need macOS's Accessibility permission, which you grant once.
  - The app asks once, at the first launch after this feature ships. It uses macOS's own prompt, which leads to System Settings.
  - Until the permission is granted, the double tap can't be noticed at all, so nothing happens. The Settings row shows **Allow…**.
  - With a key-combination shortcut instead (see Settings), pressing it without the permission opens the panel in a "Typing in other apps is off" state (Figma, "Direct dictation (draft)" 4). Its **Open System Settings** button goes straight to Privacy & Security › Accessibility.
- **Password fields and "secure keyboard entry":** macOS blocks typed events while secure input is on. That covers password fields and apps with "secure keyboard entry", such as Terminal's option of that name or some password managers. If secure input is on when the shortcut is used, dictation doesn't start and the icon shows the warning triangle for a moment.

## How it fits with the panel

| Situation | Shortcut does |
|---|---|
| Nothing running | Starts direct dictation |
| Panel open (listening, stopped, any state) | Closes the panel by the usual rules (listening ends, paused is kept), then starts direct dictation into the app in front |
| A paused panel dictation waiting (panel closed) | Starts direct dictation; the paused one is untouched and still there on the next click |
| Model getting ready | Opens the panel's Getting-ready state; use the shortcut again once it's ready |
| Microphone permission missing | Opens the panel's microphone state |
| Accessibility permission missing | With a key combination: opens the panel's "Typing in other apps is off" state. With a double tap: nothing, since the tap can't be noticed without the permission |

Clicking the menu-bar icon during direct dictation stops it, typing the last word, and doesn't open the panel. The right-click menu works as today.

The rule "the microphone runs only while the panel is open" becomes: **the microphone runs only while the panel is open or direct dictation is on, and the icon is red whenever it runs.**

## Settings

New rows under the existing ones, in the same layout: labels left, controls right.

| Label | Control |
|---|---|
| Shortcut | a shortcut field showing **Double Left ⇧** by default. Click it, then either double-tap a modifier key (Left or Right Shift, Control, Option or Command) or press a key combination; Esc cancels. A key combination needs ⌘, ⌃ or ⌥, or is a function key (F1–F19). |
| Mode | a menu: **Toggle** or **Hold to talk** |
| Accessibility | **Allowed ✓** once granted, otherwise an **Allow…** button that opens the right System Settings page |

**Double tap, precisely:**
- Two presses of Left Shift, each released, within 0.4 s.
- No other key and no mouse click in between, so Shift-typing capitals ("AB") or Shift-clicking never triggers it.
- Right Shift doesn't count, so Right Shift stays free for ordinary typing.

**One clash to know about:** JetBrains IDEs (IntelliJ, PyCharm, and others) use double Shift for Search Everywhere. The app only listens to keys and can't swallow them, so in those IDEs both would happen. A key-combination shortcut avoids it.

## Typing while modifier keys are held

In Hold to talk mode, Shift (or a key combination's modifiers) is held down while words are typed. `--typing-test` types into an AppKit text view with nothing held, with Shift held, and with ⌃⌥ held. It does this by each route: the HID tap, the session tap, and posting to the front app. In every case the text arrived exactly as sent; the system did register the held keys. So Hold to talk types word by word, like Toggle. Other apps (Chrome, Slack, Terminal) are checked by hand.

## Signing

The app is signed with the Apple Development certificate (team QF8GKSXVE2), so macOS keeps the Accessibility and microphone permissions across rebuilds. An ad hoc signature would make every build a new app.

## Build

- **`HotkeyCenter`** (app) reports "start" and "stop" from either kind of shortcut:
  - **A double tap of a modifier:** it watches modifier-key changes with a global event monitor, which needs Accessibility, plus a local one for when the app's own windows have focus. It tells Left from Right Shift by key code, and resets on any other key or a mouse click.
  - **A key combination:** it uses the system's global hotkey call (`RegisterEventHotKey`), which needs no permission.

  It re-registers when Settings changes the shortcut.
- **`DoubleTapDetector`** (Kit): the timing and reset rules above as plain logic, fed key events, so it can be tested without a keyboard.
- **`TextTyper`** (Kit protocol): `type(_ text: String)`. The app's version posts `CGEvent` keyboard events with the text, in pieces of up to 20 characters (the most one event carries). Tests use a fake that records what was typed.
- **`DirectDictation`** (Kit): starts a session on the loaded model and the microphone. It types each newly finished word as updates arrive, and on stop types the last word, archives the text (marked typed) and turns the microphone off. Toggle and Hold to talk differ only in which hotkey events start and stop it. It shares the model, the microphone queue and History with the panel's `DictationController`, and keeps to the rules in the table above.
- **Settings:** a shortcut field, a Mode menu, and an Accessibility row. They're stored in UserDefaults like the model and the history length.
- **History:** `HistoryEntry` gains `typed`, and the History window shows it as "· typed".
- **Icon and sound:** the icon turns red once the microphone is live (`DictationController.isMicrophoneLive`), not when it is asked to start. That applies to the panel too. A direct dictation also plays the system sound "Tink" at that moment.
- **Stopping keeps the last word:** every stop waits until the microphone has stopped and handed over its last audio before the session finishes.

## Tests

**Logic (fakes):**
- `DoubleTapDetector`:
  - two Left Shift taps within 0.4 s fire;
  - slower taps, Right Shift, Shift with a letter between, or a click between don't fire;
  - in Hold to talk, Shift then Shift-held (the second press within 0.4 s of the first release) fires start, and releasing it fires stop.
- Toggle: start → listening, mic on, icon red; words arrive → only finished words typed, in order, with the right spacing; stop → last word typed, mic off, a History entry marked typed.
- Hold to talk: the second press of the double tap starts, its release stops.
- Panel open + shortcut → the panel closes by its rules, then direct dictation starts. A paused panel dictation survives a direct dictation.
- The shortcut while getting ready, or with a permission missing → the panel opens in the matching state, and nothing is typed.
- Clicking the icon during direct dictation stops it.

**End to end (real model, real typing):** a new `--direct-selftest <audio>` mode opens a test window with a text field, focuses it, plays the file through direct dictation, and checks that the field holds exactly the dictation's final text. It needs the Accessibility permission.

**By hand:** the modifier-key test above, plus a real dictation into TextEdit, Chrome, Slack and Terminal in both modes.

## Order of work

1. The held-modifier test (Shift and ⌃⌥), using a throwaway typing prototype. I'll report the result, which decides how Hold to talk types.
2. `TextTyper`, `DirectDictation` and the hotkey in Toggle mode, with tests and the end-to-end self-test.
3. Hold to talk.
4. Settings rows, the permission panel state, and "typed" in History.

Each step is committed with its tests passing, and the app is rebuilt and restarted after each.
