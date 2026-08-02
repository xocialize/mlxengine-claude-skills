# Realtime-audio apps on MLXEngine (mic in, speech out)

Guidance for apps that put a **microphone** and/or **live speech playback** in front of the
engine — voice chat, dictation, companion assistants. Everything here was paid for in crashes
and re-downloads; read it before writing the mic layer, not after.

The reference implementations are `voice-chat-kit` (`VoiceChatKit`, the session/turn layer) and
`llm-voice-chat` (the proving app), plus the Nemotron ASR Demo (capture patterns) and the
Gepard Demo (streaming playback).

---

## 1. The Swift 6 realtime-tap trap ⚠️ (four crash reports)

**Symptom.** The app dies the instant recording starts — the very first audio buffer — with:

```
Thread N Crashed:: Dispatch queue: RealtimeMessenger.mServiceQueue
  _dispatch_assert_queue_fail
  dispatch_assert_queue
  _swift_task_checkIsolatedSwift
  swift_task_isCurrentExecutorWithFlagsImpl
  closure #N in MicCapture.start(...)          ← small offset: the PROLOGUE
EXC_BREAKPOINT (SIGTRAP)
```

**Cause.** `AVAudioNode.installTap(onBus:bufferSize:format:block:)` takes a
`(AVAudioPCMBuffer, AVAudioTime) -> Void` — **not `@Sendable`**. In Swift 6, a non-Sendable
closure literal *inherits the isolation of the scope that forms it*. Written inline inside a
method of a `@MainActor` type (which every `@Observable` capture class is), the tap block
becomes MainActor-isolated, and the compiler emits a dynamic executor check **in the closure's
prologue**. CoreAudio calls it on the realtime thread, the check runs `dispatch_assert_queue`,
and that traps rather than returning false.

**The tell, and the experiment.** The crash offset is *small and constant* across attempts
(`+140`, `+192`, `+196`) — that is the prologue, before any of your code. If the offset doesn't
move when you rewrite the body, the problem is the closure's **signature/isolation**, not
anything it calls. Confirm in one step: empty the tap body completely. If it still traps, stop
editing the body.

**Wrong fixes** (all tried, all still trap — the isolation attaches before the body runs):
- removing `self` capture and reading only locals;
- typing the callbacks `@Sendable` (SE-0434: a `@Sendable` closure can *still* be inferred
  global-actor-isolated, e.g. by capturing `[weak self]` of a `@MainActor` class);
- building callbacks in a `nonisolated static` factory;
- replacing callbacks entirely with an `AsyncStream` continuation.

**Right fix.** Form and install the tap from a **`nonisolated`** function, so there is no
ambient actor to inherit. Pass everything it needs as parameters; capture nothing from `self`:

```swift
private nonisolated static func installTap(
    on input: AVAudioNode, bufferSize: AVAudioFrameCount,
    inputFormat: AVAudioFormat, outputFormat: AVAudioFormat,
    converter: AVAudioConverter, ratio: Double
) -> AsyncStream<[Float]> {
    let (buffers, relay) = AsyncStream.makeStream(of: [Float].self,
                                                  bufferingPolicy: .bufferingNewest(64))
    input.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { buffer, _ in
        // …convert…
        relay.yield(samples)          // yield is lock-based and isolation-free
    }
    return buffers
}
```

Then drain the stream in a `Task { @MainActor }` and invoke app callbacks there. Declare those
callbacks `@MainActor` **outright** so isolation is stated, not inferred, and the compiler
enforces the boundary instead of the runtime trapping at it. Cost: one hop per ~100 ms buffer.

**Latent elsewhere.** The same code under **Swift 5** is merely an unchecked data race and runs
fine — which is why the pattern survives in older apps. The **Nemotron ASR Demo** has this exact
latent bug today (Swift 5 target); any Swift 6 migration of it will trap on the first buffer.

**Rule of thumb.** On a realtime audio thread: no actor-isolated closures, no `Task`-local
reads, no MLX. Copy, `yield`, return.

---

## 2. Half-duplex turn-taking, and the drain signal

Voice loops without echo cancellation must be strictly half-duplex or the assistant transcribes
itself:

- Close the mic **at the endpoint**, before generation even starts — not when audio begins.
- Re-open only after playback has *actually finished*. `AVAudioPlayerNode` has no "played out"
  event by default: schedule with
  `scheduleBuffer(_, completionCallbackType: .dataPlayedBack)` and resolve an awaitable
  `drained()` from the last callback. Without it you cannot correctly re-arm, and turn-total
  latency is unmeasurable.
- After draining, wait a small margin (~150 ms) and **re-calibrate the VAD** — room reverb of
  the assistant's final syllable will otherwise trip speech-onset immediately.

True barge-in needs `AVAudioEngine.setVoiceProcessingEnabled(true)` (AEC) and a shared graph for
capture + playback; treat it as a separate project, not a tweak.

---

## 3. Endpointing (VAD) that doesn't fight the user

Energy VAD is the right first tool — a second neural model on the inference actor to make a
threshold decision is a bad trade. What matters:

- Feed it **raw RMS**, not the display-scaled level the waveform uses.
- **Hysteresis**: speech-on at `floor × 3`, silence only below `floor × 2`, so a trailing-off
  word doesn't end the turn.
- The trailing-silence counter **is** the hangover; don't add a second mechanism.
- Adapt the noise floor **only on silent frames** — never learn the user's voice as room noise.
- Calibration blackout on `arm()` (~500 ms) — doubles as the reverb guard on re-arm.
- Min-utterance discard (coughs) and a max-utterance cap (stuck mic).
- Keep feeding **all** audio to STT during the window; the VAD gates the *endpoint*, not the
  feed, or you clip word onsets.

It's a pure state machine, so it tests offline at exact frame counts (~10 Hz from 100 ms
buffers). Do that — endpointing feel is otherwise unverifiable without a person in the room.

---

## 4. Actor serialization shapes the pipeline, not just the speed

All engine inference serializes on `@InferenceActor`. In a voice loop that means while a TTS
segment synthesizes, LLM token production **pauses** (its iterator is pull-based; the KV cache
is untouched) and resumes after. This is correct, not a bug: TTS running ~3× realtime outruns
playback, so speech stays continuous; only wall time stretches slightly.

What it *does* forbid: holding a live STT session open while TTS streams (partials stall, and
you'd capture speaker audio anyway). Sequence strictly — finish transcription, then generate
and speak.

---

## 5. Latency: measure the phases, or you'll optimize the wrong one

Emit a `[TURN]` line per turn, all offsets from **speech end** (the only t0 a person
experiences):

```
[TURN] stt=74ms firstToken=2242ms think=907ms firstAudio=5740ms total=21.7s audio=15.9s
       gen=347tok@282tok/s phys=9.87GB
```

That was the **first turn of a fresh process**, and it is not representative — see below.

**Warm up at load time, then measure.** Run a throwaway 1-token generation and a very short
synthesis *using the real reference clip* inside `activate()`. That pays Metal pipeline
compilation and the TTS reference encode (~850 ms) before anyone is waiting on them. Same app,
first turn, with warm-up:

```
[TURN] stt=107ms firstToken=156ms (llm 49ms) think=1357ms firstAudio=1747ms
```

Steady state settles at ~1.2 s to first audio. Where the time actually goes, warm:

| Term | Warm | Read |
|---|---|---|
| `stt` | 60–110 ms | free |
| `llm` (dispatch → first token) | ~50 ms | prefill is **not** a bottleneck |
| `think` | 0.6–1.4 s | dominates, on an always-reasoning model |
| TTS first chunk | ~200 ms | Gepard warm |

**Two traps this closed.** Profiling the first turn pointed at prefill and would have bought a
held-KV session-reuse project that optimizes a ~50 ms step. And measuring first-token from
*speech-end* buried the transcript wait and pipeline setup inside "the model is slow" — hence
`llmFirstTokenMs`, clocked from request dispatch. When an app's number and a CLI gate's number
differ by 10×, suspect the environment (cold caches, co-resident models), not the algorithm.

For reasoning models (LFM2.5 always thinks), keep `thinkMs` separate — otherwise "the model is
slow" hides "the model is deliberating", and the fix for each is different.

---

## 6. Code-only AppKit apps (the shell these usually live in)

Three failures that each look like "the app is broken":

- **`NSApplication.delegate` is weak.** A delegate held only by a local in `main.swift` (even
  inside `MainActor.assumeIsolated { }`) deallocates immediately; the app launches windowless
  with no callbacks. Hold it in a global.
- **No activation policy** → background agent: no dock icon, no key window. Call
  `app.setActivationPolicy(.regular)` before `run()`.
- **Window on another Space** → `isVisible == true` while the user stares at an empty desktop.
  Use `collectionBehavior = [.moveToActiveSpace, .managed]` and `orderFrontRegardless()`.

Also delete the template's `INFOPLIST_KEY_NSMainStoryboardFile` if you boot from `main.swift`,
and check `SWIFT_VERSION` on the app target — a Swift 5 app target consuming Swift 6 packages
compiles, but hides exactly the isolation problems in §1 until something else forces them up.

Mic entitlement checklist (all five, or it fails in confusing ways):
`app-sandbox`, `network.client` (weight downloads), `files.user-selected.read-write`,
`files.bookmarks.app-scope`, **`device.audio-input`**, plus
`INFOPLIST_KEY_NSMicrophoneUsageDescription`.
