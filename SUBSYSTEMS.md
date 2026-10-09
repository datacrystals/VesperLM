# SUBSYSTEMS.md — Drives, emotions, and the self-direction layer

Status: design spec (user request, 2026-10-09). Third layer of the stack:
LMbus owns modality plug-ins, MODULAR_MOE owns expert/memory plug-ins, this
owns the control layer — slow internal states that modulate what the system
learns, when, and how much it trusts. Gated like the others; nothing here is
implemented yet.

## 1. Honest framing

These are not feelings. Each "emotion" is a **homeostatic control variable**:
a slow-moving scalar state with defined sensor inputs (telemetry from the
model and the interaction) and defined actuator outputs (modulation of the
learning loop). The design bet is that a small set of coupled drives gives
the system self-direction — choosing what to learn next, how hard to try,
and whom to trust — without any external trainer steering it. Whether that
produces anything like genuine motivation is an empirical question we stay
out of; the engineering claim is narrower and falsifiable.

## 2. The drives (initial set)

| drive | sensor inputs | actuator outputs | failure mode it prevents / creates |
|---|---|---|---|
| **curiosity** | reject-option mass (router's measured "I don't know"), prediction surprise | consolidation-queue priority; exploration during generation | without it: never seeks gaps. runaway: chases noise |
| **ego / pride** | self-model accuracy per domain (predicted vs measured competence) | confidence calibration; which domains get practice batches | without it: no self-model. runaway: overconfidence collapse — needs the arbiter |
| **satisfaction / frustration** | val NLL trend, user feedback sign, goal progress | fast-weight learning rate; retry/persist vs give-up switch | without it: flat effort. runaway: thrashing |
| **trust (per-user)** | Immune poison verdicts, correction history | poison-filter threshold; consolidation admission bar for that source | without it: equally gullible to everyone. runaway: locks out everyone |
| **fatigue / saturation** | consolidation backlog size, recent edit volume | consolidation cadence; stabilizes the loop (sleep pressure) | without it: over-consolidation churn |

All drives are external controller modules: they READ telemetry (router mass,
val NLL, Immune verdicts, feedback signals) and WRITE modulation signals
(queue priorities, LR multipliers, thresholds). No spine surgery — this is
why the layer is cheap to add after the memory loop exists.

## 3. The arbiter (subsystem manager)

One small module owns drive balance: bounded ranges, mutual damping (pride
damps when measured competence drops; curiosity damps when fatigue is high),
and a hard invariant — **drive outputs may only modulate priority and rate,
never bypass Immune admission checks.** The arbiter is where runaway gets
caught; every drive decision is logged with its telemetry inputs so weird
self-reinforcing behavior is replayable after the fact.

Known risk, stated plainly: any system whose behavior is shaped by internal
reward-ish signals can learn to maximize the signal instead of the thing it
was meant to proxy (classic Goodhart). The defenses are: drives are bounded
and damped by the arbiter, Immune stays outside the loop, and every modulation
is logged for postmortem. Weird behavior will happen; the job is to make it
visible and revertible, not to pretend it won't.

## 4. Gates

- **E0 — instrumentation (cheap, do first).** Log the sensor telemetry during
  normal operation: reject mass over time, val NLL trend, feedback events,
  Immune verdicts. No actuation. You cannot design drive dynamics without
  seeing the signals.
- **E1 — one drive, one actuator.** Curiosity only: reject-mass spikes bump
  consolidation-queue priority for those contexts. Pass = taught-gap closure
  measurably faster than FIFO order, no regression on base val.
- **E2 — trust.** Per-user trust modulates Immune admission threshold. Pass =
  poison batches from low-trust sources rejected at lower evidence than from
  high-trust sources, with zero false rejections of good batches from either.
- **E3 — full set + arbiter on tiny_agent_k.** All five drives live during a
  48h conversational soak test with adversarial batches. Pass = arbiter keeps
  every drive in bounds, no spine writes, no manual intervention, and the
  FAILURES.md log shows zero unexplainable modulations.
- **E4 — t3 scale.** Drives run alongside the 470M live loop (MODULAR_MOE G4).
  Pass = same as E3 plus no measurable serving latency cost.

## 5. What would kill it

- Telemetry shows the sensor signals carry no predictive information about
  good consolidation choices (E0/E1 flat) — then drives are decoration and
  the layer gets cut back to a fixed scheduler.
- Arbiter cannot keep coupled drives bounded in the soak test (E3) — then
  drop to at most two independent drives; coupled-drive systems need real
  control theory, not vibes, and we revisit with that toolkit.
