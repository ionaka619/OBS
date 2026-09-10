# OBS Face Sticker

An OBS Studio 32 effect filter for Windows x64 that tracks one face and attaches
a transparent sticker to it. Detection runs asynchronously so it does not block
OBS's video thread. It includes glasses, heart eyes, crown, cat ears, bunny ears,
moustache, blush, and halo stickers, plus custom PNG/JPEG input.

## Use in OBS

1. Install the release package and restart OBS.
2. Add or select a **Video Capture Device** source.
3. Right-click the camera and choose **Filters**.
4. Under **Effect Filters**, click **+** and select **Face Sticker**.
5. Choose a sticker, then tune scale, offsets, smoothing, and detection interval.

Bright, front-facing video gives the most stable results. `Detect every N frames`
trades responsiveness for CPU usage; 3 is a good starting value.

## Windows build

Requirements: Visual Studio 2022 with Desktop development with C++, CMake 3.30+,
Git, and vcpkg exposed through `VCPKG_ROOT`.

From a Developer PowerShell:

```powershell
cmake --preset windows-x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build --preset windows-x64 --config RelWithDebInfo
cmake --install build_x64 --prefix release/RelWithDebInfo --config RelWithDebInfo
```

The included GitHub Actions workflow uses the same process on a Windows runner.
Push the project to GitHub, open **Actions**, choose **Windows Build**, run it,
and download the resulting artifact.

## Architecture

- `filter_video` copies supported camera frames to a single-slot worker queue.
- OpenCV Haar cascades detect the largest face and eye-line rotation.
- Exponential smoothing reduces position and rotation jitter.
- A custom OBS GPU effect alpha-composites the selected sticker.

This MVP uses a face box and eye-line rather than a dense facial mesh. It is
well suited to hats, ears, glasses, blush, and moustaches. A later MediaPipe
backend can add expression-driven animation and precise mouth/eye landmarks.

## License

Plugin code: GPL-2.0-or-later. Sticker artwork: CC0-1.0. See
`THIRD_PARTY_NOTICES.md` for OpenCV notices.
