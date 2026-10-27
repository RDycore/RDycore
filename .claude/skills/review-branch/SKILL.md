---
name: review-branch
description: Review an RDycore branch or MR against origin/main — the whole process, in the order that catches the most bugs. Use when asked to "review this branch", "review my changes", "review the MR", or to check a branch before pushing.
argument-hint: <branch | commit-ref | empty for HEAD>
---

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that is not a single
ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`. `DEST` is `origin/main`.

Run, as one shell command:

    git fetch origin --quiet && git diff --shortstat DEST...SRC && git diff DEST...SRC > /tmp/rdy-review-$$.diff && wc -l /tmp/rdy-review-$$.diff

State `SRC`, its SHA, `DEST`, the shortstat and the diff path. Stop and report if
`SRC` does not resolve or the diff is empty.

Then work @review-procedure.md in order. **Do the steps in that order**: it is
ordered by how many of the eight bug fixes in the AMR series each step would have
caught, not by convenience. Reading the diff is step 5 of 6 for a measured
reason, not a slight.

Each step may be run on its own, and three have their own skills:

| step | what | skill |
|---|---|---|
| 1 | configuration-axis coverage | `axis-coverage` |
| 2 | exactness-class invariant assertions | — (in the procedure) |
| 3 | backend-differential run | `backend-diff` |
| 4 | index-map audit | `index-map-audit` |
| 5 | read the diff | — (in the procedure) |
| 6 | verify the gate claims | `gate-claims` |

Steps 3, 4 and 6 build or run things and can take a while; steps 1, 2 and 5 are
reading. If the user wants a quick pass, do 1, 2 and 5 and say which steps were
skipped — never imply coverage that was not attempted.

Compose one report per the **Report** section of the procedure. Report to stdout
only; do not post anything anywhere.
