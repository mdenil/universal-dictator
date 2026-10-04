# Universal Dictator: direct dictation

Direct dictation adds a global shortcut that dictates straight into the text field in focus, in any app, without opening the panel. Everything the app does today stays exactly as it is: the panel, pause and resume, Copy ⌘C, History and Settings. Direct dictation is an extra way in.

The new UI is drafted in Figma under "Direct dictation (draft)": [Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl). It has three parts: the Settings window with its new rows, a panel state for the missing permission, and an overview of direct dictation in another app.

## What it does

1. **Press the shortcut** (default **⌃⌥D**) while typing in any app: the menu-bar icon turns red and the microphone starts. No window opens.
2. **Speak.** Each word is typed into the focused text field as if you had typed it, as soon as it is final. That is about 0.3 s after you start the next word. The word still being recognised is never typed, so nothing ever needs to be erased and retyped.
3. **Stop**, in one of two modes chosen in Settings:
   - **Toggle** (default): press the shortcut again.
   - **Hold to talk**: hold the shortcut while speaking; releasing it stops.

   Stopping types the last word, turns the microphone off and the icon back to normal.
4. The dictation goes to History, marked **typed**, so the text is never lost even if it landed in the wrong place.

The text goes wherever keyboard focus is as each word arrives. Clicking into another field mid-dictation sends the rest there, just as typing would.

## How typing works

The app sends each word as keyboard events carrying the text (`CGEvent` with a Unicode string). To the receiving app it looks like fast typing, so it works in native apps, browsers, Electron apps such as Slack, and terminals. Undo in the receiving app removes the typed text the way it removes typing.

- **Spacing:**
  - Words are typed with the spacing the model gives them.
  - The first word of a dictation gets no leading space, so the text starts exactly at the cursor.
  - Each later word starts with a space.
- **Accessibility permission:** sending keystrokes to other apps needs macOS's Accessibility permission, which you grant once. Until it's granted, pressing the shortcut opens the panel with a "needs permission" state (Figma, panel state 6). It has an **Open System Settings** button that goes straight to Privacy & Security › Accessibility.
- **Password fields and "secure keyboard entry":** macOS blocks typed events while secure input is on. That covers password fields and apps with "secure keyboard entry", such as Terminal's option of that name or some password managers. If secure input is on when the shortcut is pressed, dictation doesn't start and the icon shows the warning triangle for a moment.

## How it fits with the panel

| Situation | Shortcut does |
|---|---|
| Nothing running | Starts direct dictation |
| Panel open (listening, stopped, any state) | Closes the panel by the usual rules (listening ends, paused is kept), then starts direct dictation into the app in front |
| A paused panel dictation waiting (panel closed) | Starts direct dictation; the paused one is untouched and still there on the next click |
| Model getting ready | Opens the panel's Getting-ready state; press the shortcut again once it's ready |
| Microphone or Accessibility permission missing | Opens the panel's permission state |

Clicking the menu-bar icon during direct dictation stops it, typing the last word, and doesn't open the panel. The right-click menu works as today.

The rule "the microphone runs only while the panel is open" becomes: **the microphone runs only while the panel is open or direct dictation is on, and the icon is red whenever it runs.**

## Settings

New rows under the existing ones, in the same layout: labels left, controls right.

| Label | Control |
|---|---|
| Shortcut | a shortcut field showing **⌃⌥D**. Click it, then press the new keys; Esc cancels. A shortcut needs ⌘, ⌃ or ⌥, or is a function key (F1–F19). |
| Mode | a menu: **Toggle** or **Hold to talk** |
| Accessibility | **Allowed ✓** once granted, otherwise an **Allow…** button that opens the right System Settings page |

**Your call: the default shortcut.**
- **⌃⌥D (recommended):** D for dictate, and as far as I know nothing in macOS uses it.
- **⌃⌥Space:** easier to hit, but macOS uses it to switch keyboard input sources when more than one is set up.
- **Fn (Globe):** macOS's own dictation key. Catching it needs a different mechanism that macOS treats as monitoring the keyboard, so it means a second permission. Not in the first version.

## Open question to settle in the first step: holding modifier keys while typing

In Hold to talk mode, ⌃ and ⌥ are physically held down while words are being typed. If a receiving app reads which keys are held, rather than the keys carried in the typed event, a typed "s" could act as ⌃⌥S, a shortcut. I don't know which apps do this. It has to be tested before anything else is built.

- **Test:** with ⌃⌥ held, type into TextEdit, Chrome, Slack, Notes and Terminal, and check the text arrives as plain text.
- **If it's safe:** Hold to talk types word by word, like Toggle.
- **If it isn't:** in Hold to talk mode the words are held back and typed the moment you release the shortcut. That's still quick, but they don't appear while you speak. I'll report back before choosing.

## A development snag: permission after every rebuild

The app is signed ad hoc, so macOS treats every build as a new app, and the Accessibility permission doesn't carry over to a rebuilt app. While I'm building this, each new build will need the permission granted again (switched off and on). Released builds signed with a real certificate keep it.

## Build

- **`HotkeyCenter`** (app): registers the shortcut with the system's global hotkey call (`RegisterEventHotKey`), which needs no permission, and reports presses and releases. It re-registers when Settings changes the shortcut.
- **`TextTyper`** (Kit protocol): `type(_ text: String)`. The app's version posts `CGEvent` keyboard events with the text, in pieces of up to 20 characters (the most one event carries). Tests use a fake that records what was typed.
- **`DirectDictation`** (Kit): starts a session on the loaded model and the microphone. It types each newly finished word as updates arrive, and on stop types the last word, archives the text (marked typed) and turns the microphone off. Toggle and Hold to talk differ only in which hotkey events start and stop it. It shares the model, the microphone queue and History with the panel's `DictationController`, and keeps to the rules in the table above.
- **Settings:** a shortcut field, a Mode menu, and an Accessibility row. They're stored in UserDefaults like the model and the history length.
- **History:** `HistoryEntry` gains `typed`, and the History window shows it as "· typed".
- **Icon:** red while direct dictation listens, the same as panel listening, with no new shape or colour.

## Tests

**Logic (fakes):**
- Toggle: press → listening, mic on, icon red; words arrive → only finished words typed, in order, with the right spacing; press → last word typed, mic off, a History entry marked typed.
- Hold to talk: press starts, release stops.
- Panel open + shortcut → the panel closes by its rules, then direct dictation starts. A paused panel dictation survives a direct dictation.
- The shortcut while getting ready, or with a permission missing → the panel opens in the matching state, and nothing is typed.
- Clicking the icon during direct dictation stops it.

**End to end (real model, real typing):** a new `--direct-selftest <audio>` mode opens a test window with a text field, focuses it, plays the file through direct dictation, and checks that the field holds exactly the dictation's final text. It needs the Accessibility permission.

**By hand:** the modifier-key test above, plus a real dictation into TextEdit, Chrome, Slack and Terminal in both modes.

## Order of work

1. The modifier-key test, using a throwaway typing prototype. I'll report the result, which decides how Hold to talk types.
2. `TextTyper`, `DirectDictation` and the hotkey in Toggle mode, with tests and the end-to-end self-test.
3. Hold to talk.
4. Settings rows, the permission panel state, and "typed" in History.

Each step is committed with its tests passing, and the app is rebuilt and restarted after each.
