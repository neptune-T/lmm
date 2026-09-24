# Isaac Sim execution blocker

Recorded 2026-09-24. This blocks the live half of
`ALPHABOT2_ISAAC_EMBODIMENT_SMOKE_PASS`.

## No reachable machine can run Isaac Sim

| Host | GPU | Disk free | Isaac Sim | Status |
|---|---|---|---|---|
| Local `plote` workstation | RTX 3070 Laptop, 8 GB | **2.9 GB** | not installed | **infeasible** — Isaac Sim needs ~30 GB |
| A800 host (4× A800 80 GB) | suitable | ~2.5 TB | unknown | **unreachable** |

### Local

Disk is at 100 % with 2.9 GB free on a 916 GB volume. Isaac Sim's installed
footprint is roughly an order of magnitude larger, before USD assets. The GPU
is also at the bottom of the supported range. Not a candidate.

### A800 — SSH access is broken

The jump-host route recorded in project memory no longer authenticates.

```
local → jumper (100.74.253.40, Windows OpenSSH 9.5) → A800 (180.76.138.216:12)
```

Observed:

* The host key **matches** `known_hosts` — it is the same machine, not a
  reassigned address.
* Port 22 is **intermittently reachable** (4 of 6 probes timed out, 2 connected).
* When it does connect, authentication fails **every time** (8/8 attempts):

```
debug1: Offering public key: ~/.ssh/id_rsa   RSA     SHA256:AaFYieIbbpqBibIfD7tau7k2NyZxqYg3Lfz2ai5+9tE
debug1: Offering public key: ~/.ssh/id_ed25519 ED25519 SHA256:aEfs87AD+vcQpD4j5p7Gchk07zQlbTwrpcE8dDndue4
debug1: Authentications that can continue: publickey,password,keyboard-interactive
→ Permission denied (publickey,password,keyboard-interactive)
```

Both keys are offered and both are rejected. This is an authorization change on
the jump host, not a network fault and not a client misconfiguration.

### Likely cause and fix (needs the operator)

The workstation's public key is no longer in the jump host's authorized keys.
On Windows OpenSSH this commonly breaks because **administrator accounts read
`C:\ProgramData\ssh\administrators_authorized_keys`**, not
`C:\Users\<user>\.ssh\authorized_keys` — an edit to the latter silently has no
effect for an admin user.

To restore, on the jump host add this line to the correct file:

```
ssh-ed25519 AAAA... plote@workstation     # SHA256:aEfs87AD+vcQpD4j5p7Gchk07zQlbTwrpcE8dDndue4
```

and ensure the file's ACL grants only `SYSTEM` and `Administrators`.

The intermittent port-22 timeouts are a separate issue (the address is in the
Tailscale CGNAT range, so the tunnel appears to be flapping) and should be
checked as well.

## What this blocks, and what it does not

**Blocked** — every item needing a running simulator: robot load, camera
render, joint/gripper/base actuation, atomic-action timing, live temporal
measurement, snapshot/restore, navigation qualification.

**Not blocked, and completed** — asset provenance, the full static kinematic
audit, the policy contract, the transform stack, the Isaac adapter's structure,
the π0.5 checkpoint audit, and all CPU tests.

## Not worked around

No robot geometry was fabricated, no numbers were invented, and no simulator
result is reported as measured. The stop conditions call for reporting a
blocker rather than improvising, and the temporal/gripper/base values that
would otherwise be guessed are exactly the ones that caused defects in the
previous WXAI embodiment.
