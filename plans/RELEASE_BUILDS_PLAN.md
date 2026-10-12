# Universal Dictator: release builds on GitHub

**The goal:** pushing a release tag, for example `v0.2.0`, makes GitHub build the Mac app and publish it on the repository's Releases page as a download. `dictate-cli` is not built or published: it stays build-it-yourself from the repository.

Your decisions: signed ad hoc for now; builds only for version tags; a DMG.

## What a release looks like

1. **You tag and push:**
   ```sh
   git tag -a v0.2.0 -m "Universal Dictator 0.2.0" && git push origin v0.2.0
   ```
2. **GitHub Actions builds the app** from that commit, on a macOS runner, in about 12 minutes (see "How long it takes"). Nothing from your Mac is used.
3. **The release appears** on the Releases page, with:
   - `UniversalDictator-0.2.0.dmg`: the app, about 1.1 GB;
   - its SHA-256;
   - notes generated from the commits since the previous tag, which you can edit afterwards.

**The DMG:**
- Opening it shows a window with the app, an Applications shortcut to drag it onto, and a Licenses folder.
- It has no custom background picture. That needs scripting Finder, which is unreliable on a headless build machine; it can be added later.

**Version numbers:**
- The app's version comes from the tag: `v0.2.0` becomes 0.2.0, shown in Finder's Get Info.
- The build number is the workflow's run number.
- The project's own `MARKETING_VERSION` (0.1 today) only matters for local builds.

**Pre-releases:** a tag with a suffix, such as `v0.2.0-rc1`, is published marked as a pre-release.

**Visibility:** while the repository is private, only you can see and download releases. Once it's public, anyone can.

## How the app gets built on GitHub

The app needs its model folder (`build/app-assets`). GitHub's runner makes it from nothing, exactly as the README's "Building the app" section does:
1. download Nemotron's checkpoint at the pinned revision;
2. export it;
3. build the Core ML encoder;
4. `dictate-cli convert app-assets`;
5. `xcodebuild`.

The build is therefore reproducible from the tag alone. I checked that the Core ML encoder build is deterministic: running it again gave byte-identical weights.

**The runner:** `macos-26` (Apple silicon, M1, 3 cores, 7 GB of memory, 14 GB of disk). Its Xcode 26.5 is the version you build with locally, and the workflow selects it explicitly. `uv` isn't installed on the image; the workflow installs it with `astral-sh/setup-uv`.

**Memory and disk are tight but fit:**
- **Memory:**
  - Measured on this Mac, the checkpoint export peaks at 5.1 GB resident and the Core ML build at 6.1 GB (3.4 GB actual footprint).
  - Most of that is memory-mapped weight files, which macOS can page out, so it should fit in 7 GB.
  - If the first run runs out of memory, the fallback is to build the three Core ML functions in separate processes. That needs one small change to `tools/nemotron_ane.py`.
- **Disk:**
  - The workflow downloads only the `.nemo` file (2.5 GB), not the duplicate `model.safetensors`.
  - It deletes each intermediate once the next step has used it.
  - That keeps the peak at about 8 GB.

**Caching:**
- The finished model folder (1.2 GB) is cached, keyed on everything that produces it:
  - `tools/`;
  - the pinned revision;
  - the DictatorCore sources;
  - `dictate-cli`'s source.
- A release that changes none of those skips straight to `xcodebuild` (about 5 minutes).
- GitHub drops caches unused for 7 days, so with infrequent releases most builds will start cold. That's fine, just slower.

## Signing: ad hoc for now

The app is signed ad hoc, as local builds are.

**For people downloading it:**
- macOS 15 and later refuse to open a downloaded ad hoc app at first. The user goes to System Settings › Privacy & Security and clicks **Open Anyway**, once per release. Right-click › Open no longer works.
- Each release is a new identity to macOS, so it asks for the microphone and Accessibility permissions again after every update.
- The README and the release notes explain both.

**Moving to Developer ID and notarization later:**
- It needs the paid Apple Developer Program, a Developer ID Application certificate and an App Store Connect API key (five repository secrets).
- It also needs the hardened runtime, with the microphone entitlement.
- After that, downloads open normally and keep their permissions across updates.
- It's a few lines in the same workflow, not a redesign.

## Licenses inside the download

The app carries Nemotron's weights, so publishing it is redistributing an NVIDIA model. The NVIDIA Open Model License requires:
- a Notice text file reading "Licensed by NVIDIA Corporation under the NVIDIA Open Model License";
- a copy of the agreement for the recipient.

The app will carry these in `Contents/Resources/Licenses/`, and the DMG shows the same files in its Licenses folder:
- `LICENSE` (Apache-2.0) and `NOTICE`. `NOTICE` gains the NVIDIA sentence.
- `NVIDIA-Open-Model-License.txt`. It's saved in the repo, as `docs/licenses/`, so every build includes the same text.

The release notes link to them. A copy of the app built locally gets them too, since it's the same build.

## When it builds

Only when you push a version tag (`v` followed by a version). There are no builds on ordinary pushes and no manual button.

## How long it takes, and what it costs

**Time:** measured on this Mac (M3, 8 cores) from cold:

| Step | Time on this Mac |
|---|---|
| Checkpoint export | 7 s |
| Core ML encoder build | 91 s |
| `dictate-cli` build | 20 s |
| App build | 26 s |
| DMG | 35 s |

GitHub's runner is an M1 with 3 cores, so I'd expect the compute steps to take 2–3 times as long, about 8 minutes. Add about 4 minutes for:
- starting the runner;
- installing the Python environment;
- downloading the 2.5 GB checkpoint;
- uploading the 1.1 GB DMG.

So about **12 minutes** for a cold build, and about **6** when the model folder comes from the cache. The first real run will show the actual figure.

**Measured:** the trial release `v0.1.0-rc1` took 5 min 17 s from cold, on its first run: model folder 2 min 44 s, app 33 s, DMG 54 s, publishing 42 s. The runner was faster than I estimated.

**Cost:**
- **Private repository:** the free plan includes 2,000 Actions minutes a month. macOS minutes count ten times, so that is about 200 macOS minutes, roughly 15 releases a month. Beyond that, macOS time costs $0.062 a minute (about 75¢ a release).
- **Public repository:** free.

## Not included

- **Automatic updates inside the app** (Sparkle or similar). People download new releases by hand.
- **A custom background picture for the DMG window.**
- **Intel Macs.** The app is Apple-silicon only, as it is now.
- **`dictate-cli`.**

## Order of work

1. **Licenses in the app:** the NVIDIA agreement text, the `NOTICE` line, and the build phase that copies them into the bundle. Checked locally.
2. **The workflow** (`.github/workflows/release.yml`):
   - the model-folder steps and their cache;
   - the versioned `xcodebuild`;
   - the DMG, made with `hdiutil`;
   - publishing the release with `gh release create`.
3. **A trial release:** `v0.1.0-rc1`, published as a pre-release. Then on this Mac:
   - download it with `gh`;
   - mark it quarantined, as a browser download would be;
   - mount the DMG and copy the app out;
   - check that macOS asks for Open Anyway as expected;
   - run `--selftest` on it.

   Afterwards, delete the trial release and tag, or keep them, as you prefer.
4. **README:** a "Download" section at the top, saying where releases are and what to expect on first open (Open Anyway, the permissions), plus how to cut a release.

Each step is committed and pushed. Pushing the workflow file builds nothing until a tag arrives.
