# Design vocabulary — adopt DesignScaffold before you write any UI

**Read this before the first SwiftUI view of a new app, not after.** Retrofitting a token
layer means touching every view you already wrote; adopting it first costs one package
reference. Every demo app in the fleet that skipped this step re-derived spacing, radii,
and type by eye and drifted — from each other and from the platform. That drift is the
reason DesignScaffold exists.

## The authority rule (AB-D-0042)

`DesignScaffold` is the fleet's **single design authority**. Colour, type, spacing, radii,
and layout come from `DesignScaffold.Tokens`. No app or package defines a second token set.

**No view hardcodes a colour, font size, or spacing value.** If the value you need is
missing, it gets **added to `Tokens`** (by a bridge ask to the `DesignScaffold` area, shipped
same-day as a patch release) — never invented locally. That constraint is what keeps a
Figma refresh a one-file change instead of a fleet-wide hunt.

## Where each layer comes from

| Layer | Source |
|---|---|
| Design vocabulary — colour, type, spacing, radii, `cardSurface()` | **DesignScaffold** (authoritative) |
| Shared components — calendar, playlist, loading modal, run stepper | **DesignScaffold** (selectable products) |
| Engine-management panels — `EngineSettingsView`, `ModelStorageSettingsView`, `ModelStateView` | **MLXEngineUI** — reuse, never rebuild |
| App-specific product UI — chat views, editors, composites | **the app** |

## Check the catalog first

**`DesignScaffold/Docs/COMPONENTS.md`** is the canonical, generated catalog — what exists,
how to adopt it, and how to request something new. It is regenerated from `Package.swift`
in the same change that ships a component, so it cannot drift.

Adopt:

```swift
.package(url: "https://github.com/xocialize/DesignScaffold.git", from: "0.1.0")
// select ONLY the products you use:
.product(name: "DesignScaffold", package: "DesignScaffold")            // tokens
.product(name: "DesignScaffoldLoading", package: "DesignScaffold")     // loading modal
```

Every component product re-exports `DesignScaffold`, so one import brings `Tokens` and
`cardSurface()` along. Each wears the scaffold look **by default** — no theme call at the
call site — and exposes a `*Theme` whose initializer defaults ARE the token values, so a
custom theme that only overrides colours still inherits the scaffold geometry.

```swift
import DesignScaffold

Text("Real-time factor")
    .font(Tokens.Font.caption)
    .foregroundStyle(Tokens.Color.secondaryLabel)
    .padding(Tokens.Space.m)
    .cardSurface()
```

### Components that matter to an engine-consuming app specifically

- **`DesignScaffoldLoading`** — `LoadingCard` + `.loadingModal(isPresented:progress:title:)`
  for model load. It is display-only and engine-agnostic: map the engine's phases into
  `LoadingProgress(fraction:status:fields:)` yourself (download → prepare/compile → warmup).
  Pairs with, does not replace, `ModelStateView`.
- **`DesignScaffoldStageStepper`** — `StageStepper` for any multi-phase run. Encodes a
  measured rule worth internalising: **never synthesise a percentage across phases** —
  phases are wildly unequal in time, so a step-counted bar races and then appears frozen.
  Pulse on event arrival, render counters as text, and give a quiet slow phase an elapsed
  timer. The component exposes no overall fraction, deliberately.

## Requesting a component

If you need something that does not exist and more than one app would use it, do NOT build
a private copy — open a bridge ask to the `DesignScaffold` area:

```bash
bridge ask --to DesignScaffold --title "Propose <Name> into DesignScaffold" --body "..."
```

The intake bar (all four): the **shape has settled** through repeated real use, not one
screen · it is **already data-driven and dependency-free** (no engine/model types in the
view; hosts pass values in) · **no new tokens needed**, or name exactly which are missing ·
there is a **written contract to generalise against** (measured behaviour beats a
description of the pixels). A fifth signal short-circuits debate: **two independent apps
want it** — if the catalog's candidates list already names what you need, say so on its ask.

Build app-specific composites locally; propose anything reusable.

## Known caveat while it lasts

`MLXEngineUI` currently carries its own `MarqueeTokens.swift` (hardcoded hex), so an app
using both libraries renders two vocabularies in one window — near-identical in dark, but
in light the engine panels stay a dark island because hex cannot adapt. Conformance is
requested in **AB-A-0019** (non-breaking: the Marquee symbols become deprecated forwarders
onto `Tokens`). Until that ships: build **your** surfaces from `Tokens`, still reuse the
engine panels rather than rebuilding them, and expect the seam.
