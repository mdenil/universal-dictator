# Universal Dictator: windows, History, shortcuts, and typing into a fixed place

This plan covers eight items of feedback. Items 1–7 are fixes and small changes. Item 8 is an investigation with a prototype, and its result decides what gets built. Questions where your answer changes the build are marked **Your call**; each has a recommendation.

The visible changes are drafted in Figma: [Universal Dictator — App UI](https://www.figma.com/design/RYBeO9fxHFk9C7WMtVS6Tl), frame "Round 3 (draft)".

## 1. History and Settings behave like ordinary app windows

**Why they misbehave today:** both windows are floating panels that never make the app active, and the app has no Dock icon. Once another window covers one, there's no Dock icon to click, no ⌘Tab entry, and no Window menu to bring it back. Keyboard focus in them is also unreliable, because the app is never the active app.

**New behaviour:**
- **Opening Settings or History** (right-click menu, or ⌘,) makes Universal Dictator the active app and brings the window to the front, like any app.
- **While either window is open,** the app has a **Dock icon** and appears in **⌘Tab**, and the menu bar shows its menus:
  - the app menu, with Settings… and Quit;
  - **Edit**, so ⌘C, ⌘V and ⌘A work in History's editor and in the search field;
  - **Window**, which lists the open windows.
- **Clicking the Dock icon,** or choosing the app in ⌘Tab, brings the windows back.
- **When the last of the two windows closes,** the Dock icon goes away again and the app is back to being menu-bar only.
- **The drop-down panel is unchanged:** it still never activates the app, so dictating never takes the focus from the app you're typing in.

**Two side effects to know about:**
- While one of these windows is in front, the menu bar shows the app's own menus, so on a crowded notched MacBook the status icons may get pushed further left. That's normal for any active app.
- The Dock icon is a generic app icon until the app has an icon of its own. An icon is still on the to-do list from the App Store discussion.

**Build:**
- `MenuBarPanel.window` makes ordinary `NSWindow`s (titled, closable, miniaturizable; History also resizable), with `NSApp.activate()` on opening.
- A small `WindowPresence` helper switches the app's activation policy to `.regular` when the first of the two windows opens and back to `.accessory` when the last closes, using `windowWillClose`.
- The SwiftUI app keeps its standard menus, which supply Edit and Window.

**Test:** by hand:
- open each window, cover it with another app's window, and bring it back from the Dock and ⌘Tab;
- copy and paste in History's editor;
- close both windows and check the Dock icon goes away.

## 2. History: Copy, Edit and Delete in a column

Each entry's buttons stack top to bottom, right-aligned, instead of side by side, so the text gets more width:

```
Today at 09:10 · Fast · copied            [ Copy ]
Hi Sarah, thanks for sending the draft      Edit
over. I'll take a look this afternoon…     Delete
```

While editing, the column shows **Done** and **Cancel** in the same places. A third slot stays empty, so the column, and with it the row, keeps its height.

## 3. History: editing doesn't move anything

**Today:** Edit swaps the text for an editor of a fixed minimum height, with padding and a thicker border, so the row, and everything below it, jumps.

**New:** the text itself becomes editable in place, in the same font, at the same position, with the same line breaks. The editor grows and shrinks with the text, like the text did. The only visible change is a thin blue outline drawn *around* the text, which takes up no space. Nothing else on the screen moves.

**Build:** a text field that grows with its text (`TextField(axis: .vertical)`), in the same font with no padding of its own. The outline is an overlay, which doesn't affect layout.

**Test:** a snapshot of an entry before and during editing, checking that the row's height and the text's position are identical. Also by hand.

## 4. History: no footer note

The note "Saved on this Mac · keeps the last 50 · change in Settings" is removed.

## 5. Settings: "Push to talk"

The second mode is called **Push to talk** everywhere in the app. It was "Hold to talk". How it works doesn't change: Shift, Shift-hold while talking, release. The README and the plan use the new name too.

## 6. Settings: shortcuts can be a double tap or a key combination

**Today:** there's one field that takes either a double tap or a key combination, depending on what you press. A combination needs ⌘, ⌃ or ⌥, so Shift-A is refused, and with the focus problem in item 1 the field often doesn't get your keys at all.

**New:** the Shortcut row offers the two kinds as an explicit choice:

| Label | Controls |
|---|---|
| Shortcut | a menu: **Double tap** or **Key combination** |
| (Double tap) | a menu of keys: Left ⇧, Right ⇧, Left ⌃, Right ⌃, Left ⌥, Right ⌥, Left ⌘, Right ⌘ |
| (Key combination) | a field: click it, press the combination, and it shows, for example, **⌃A**; Esc cancels |

- **A double tap of any modifier works,** for example double-tap Left ⌘.
- **A key combination is any key plus at least one modifier,** for example ⌃A, ⌘⇧D or ⌥Space. A function key (F1–F19) also works on its own.

**Your call: Shift plus a letter (⇧A).** macOS delivers a registered shortcut to the app that registered it instead of to the app you're typing in. So ⇧A as the shortcut would mean you could never type a capital A anywhere while the app runs. Likewise ⌘A would take over Select All everywhere, and ⌘C over Copy.
- **Allow them, with a warning (recommended):** ⇧A, ⌘A and other combinations that are common typing or editing keys are accepted, but the field shows a one-line warning, for example "⇧A replaces typing a capital A everywhere". You decide.
- **Refuse them:** combinations that block ordinary typing aren't accepted.

**Build:**
- **Settings:** `ShortcutKindPicker` and a double-tap key menu replace today's one-field recorder. The combination field becomes a proper recorder that is focused when clicked. It works now that the window gets keyboard focus (item 1).
- **Kit:** a list of common typing and editing combinations, for the warning.

**Tests:**
- logic: the kinds and display names, and which combinations get a warning;
- by hand: recording ⌃A, ⇧A and ⌘⇧D, and a double tap of Left ⌘, then each one starting and stopping a direct dictation.

## 7. Figma

The "Round 3 (draft)" frame shows:
- History with the button column and no footer;
- an entry before and during editing, side by side, at the same size;
- Settings with the Shortcut kind menu, the double-tap key menu, the combination field (recording, set, and with a warning), and the Mode menu reading Toggle / Push to talk.

The existing drafts are updated to match once you approve.

## 8. Investigation: keep typing into the same place while you move around

**What you want:** start a direct dictation with the cursor in, say, Notes. Then switch to other apps while talking, and the words keep going into Notes, at the same place.

**Why this is hard:** today the app types the way a keyboard does: macOS delivers the keystrokes to whatever app and field are in focus. Typing somewhere else means not typing as a keyboard. There are three ways to do that, each with limits I can't settle without trying them.

| Approach | How | Expected to work in | Expected not to work in |
|---|---|---|---|
| **A. Send keystrokes to one app** | Post the keyboard events to the target app's process (`postToPid`) instead of the focused one | Some Cocoa apps (Notes, TextEdit) may accept them while in the background | Apps that ignore input when inactive; probably terminals and browsers. Unknown until tested |
| **B. Insert into the text field directly** | Remember the focused text field at the start (Accessibility, `AXUIElement`), then insert each word at its cursor (`kAXSelectedTextAttribute`) | Native text fields and editors (Notes, TextEdit, Mail, Messages), whether or not they're in front | Terminals (their text isn't editable this way), and browsers and Electron apps unevenly |
| **C. Hold the words until you're back** | While the target app isn't in front, keep the words. When you return to it, type them all at once | Everywhere, terminals included, because it's ordinary typing again | Nothing, but the words appear when you come back, not while you're away |

My recommendation: combine them. Use **B** where the target field accepts it, which gives live words even in the background. Use **C** everywhere else, so nothing is lost and terminals work too. A is only worth it if the test shows a clear gain over B.

**The investigation (first step, before building any of it):** a test mode that opens a scratch document in each target app and puts our own test window in front, so the target is in the background. It then tries A and B and checks what arrived:
- TextEdit and Notes;
- Chrome (a text box on a local test page);
- Slack;
- Terminal and Ghostty, using a scratch shell that discards input.

It needs the Accessibility permission the app already has. Each run takes the focus for a few seconds, so I'll tell you before running it. The result is a table like the one above with **measured** cells instead of expected ones. With that, I'll write the plan for the feature itself (C plus whichever of A or B works) for you to review.

**Questions for later (after the investigation):**
- Should this always happen, or be a setting ("Type into: the app in front / the app where I started")?
- With approach C, if you never return to the target app, should the words go to History only (my suggestion), or be pasted when you stop?

## Order of work

1. Windows (1) and the History changes (2–4), then "Push to talk" (5).
2. Shortcut kinds (6).
3. Figma (7): the draft first, so you can review it alongside this plan; the rest after approval.
4. The investigation (8), then its plan for review.

Each step is committed with its tests passing, and the app is rebuilt and restarted after each.
