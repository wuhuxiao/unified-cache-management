---
name: vllm-ucm-skill-maintainer
description: Continuously improve this repository's Codex development skills from branch commits, debugging sessions, tests, code reviews, and user feedback. Use when asked to update, refine, synchronize, or create project skills such as vllm-ucm-fawa-dev or vllm-ucm-llmperf.
---

# vLLM UCM Skill Maintainer

Use this skill to keep project development skills accurate as the codebase evolves. The goal is to capture durable engineering knowledge, not to record a changelog.

## Sources

- Prefer repository skills under `.codex/skills/` as the source of truth.
- Check for same-name user-level mirrors under `/root/.codex/skills/` when a skill is active in both places. If a mirror exists and the user expects it to be usable immediately, keep it in sync with the repo copy.
- Use current branch evidence:
  - `git branch --show-current`
  - `git log --oneline --decorate --max-count=20`
  - `git log --since=<date> --oneline --stat -- <relevant paths>`
  - `git show --stat <commit>` and targeted `git show <commit> -- <paths>`
  - current `git diff` and focused tests.

## Update Workflow

1. Identify the target skill(s) from the user's request and changed files. For FAWA/HMA work, start with `.codex/skills/vllm-ucm-fawa-dev/SKILL.md`; for llmperf validation, use `.codex/skills/vllm-ucm-llmperf/SKILL.md`.
2. Read only the relevant skill body and the relevant code/test diffs. Do not bulk-load unrelated references.
3. Extract durable rules:
   - invariants that future code must preserve,
   - edge cases that caused bugs,
   - correct lifecycle order,
   - required test or validation commands,
   - stable file paths and diagnostic grep patterns.
4. Reject noisy content:
   - temporary fix history,
   - vague "remember to be careful" advice,
   - one-off logs, local run artifacts, or transient ports,
   - large code snippets that can be rediscovered from source.
5. Patch the skill concisely. Prefer updating existing sections over appending a new section for every session.
6. Update `agents/openai.yaml` only when the skill's user-facing purpose or default prompt changes. Keep UI fields short and quote string values.
7. If syncing a user-level mirror, copy from the repo skill after patching and verify the files match.

## What To Capture

- **Invariants:** rules that should block unsafe implementation choices.
- **Workflow:** correct order of scheduler/worker calls, server start/stop, test phases, or CI inspection.
- **Diagnostics:** exact grep patterns, log files, or commands that shorten future debugging.
- **Validation:** commands that should be run after touching the relevant module.
- **Environment assumptions:** accelerator requirements, model paths, ports, or platform differences, only when stable.
- **Failure modes:** concrete symptoms and the likely code path to inspect.

## Writing Guidelines

- Keep `SKILL.md` lean. Add a reference file only if the details are too long or variant-specific.
- Use present-tense operational guidance. Avoid commit-message style prose.
- Mention commit hashes only when they identify a stable lesson that would otherwise be hard to trace.
- Prefer bullets with direct instructions over narrative explanations.
- Do not include secrets, customer data, private URLs, huge logs, or generated result files.
- Do not overwrite unrelated user edits. If a skill file is already dirty, inspect the diff and merge with it.

## Validation

After updating skills:

```bash
git diff -- .codex/skills
git status --short
```

If a user-level mirror was updated:

```bash
cmp -s .codex/skills/<skill-name>/SKILL.md /root/.codex/skills/<skill-name>/SKILL.md && printf 'skills match\n'
```

If the skill update also mentions new test commands, run or at least syntax-check those commands when practical.
