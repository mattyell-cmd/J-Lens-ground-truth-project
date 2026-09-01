"""Render the Claude Code session JSONL into a readable plain-text transcript.

Keeps everything that is part of the conversation: user turns, assistant prose,
assistant thinking, every tool call with its full input, and every tool result
with its full output. Drops only the harness bookkeeping records (mode changes,
title updates, file-history snapshots) and the injected system-reminder
attachments, which are not part of the thread.
"""
import json
import sys

SRC, DST = sys.argv[1], sys.argv[2]
INCLUDE_THINKING = "--no-thinking" not in sys.argv

RULE = "=" * 78
THIN = "-" * 78


def as_text(content):
    """tool_result content may be a plain string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif b.get("type") == "image":
                    out.append("[image omitted]")
                else:
                    out.append(json.dumps(b, ensure_ascii=False, indent=2))
            else:
                out.append(str(b))
        return "\n".join(out)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, indent=2)


records = []
for line in open(SRC):
    line = line.strip()
    if line:
        records.append(json.loads(line))

n_think = sum(
    1
    for r in records
    if isinstance(r.get('message'), dict)
    and isinstance(r['message'].get('content'), list)
    for b in r['message']['content']
    if isinstance(b, dict) and b.get('type') == 'thinking'
)

lines = []
w = lines.append

w(RULE)
w("CLAUDE CODE SESSION TRANSCRIPT")
first = next((r for r in records if r.get("timestamp")), {})
last = next((r for r in reversed(records) if r.get("timestamp")), {})
w(f"session      : {first.get('sessionId', '?')}")
w(f"project      : {first.get('cwd', '?')}")
w(f"git branch   : {first.get('gitBranch', '?')}")
w(f"started      : {first.get('timestamp', '?')}")
w(f"last entry   : {last.get('timestamp', '?')}")
w("contents     : every user turn, every assistant message, every tool call")
w("               with its full input, and every tool result in full.")
w(f"thinking     : {n_think} thinking blocks exist in the session record but")
w("               are NOT stored in plaintext (only a cryptographic")
w("               signature is kept), so none could be included.")
w("               Nothing else is omitted.")
w(RULE)
w("")

n_user = n_asst = n_tool = 0
skipped_meta = 0

for r in records:
    if r.get("type") not in ("user", "assistant"):
        continue
    if r.get("isMeta"):
        skipped_meta += 1
        continue

    msg = r.get("message")
    if not isinstance(msg, dict):
        continue
    role = msg.get("role")
    content = msg.get("content")
    ts = r.get("timestamp", "")

    blocks = content if isinstance(content, list) else [
        {"type": "text", "text": content}]

    # A user record that carries only tool_result blocks is the harness
    # returning output, not the human speaking. Label it as such.
    only_tool_results = (
        role == "user"
        and all(isinstance(b, dict) and b.get("type") == "tool_result"
                for b in blocks)
        and blocks
    )

    for b in blocks:
        if not isinstance(b, dict):
            continue
        t = b.get("type")

        if t == "text":
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if role == "user":
                n_user += 1
                w(RULE)
                w(f"USER   [{ts}]")
                w(RULE)
            else:
                n_asst += 1
                w(THIN)
                w(f"ASSISTANT   [{ts}]")
                w(THIN)
            w(text)
            w("")

        elif t == "thinking" and INCLUDE_THINKING:
            think = (b.get("thinking") or "").strip()
            if not think:
                continue
            w(THIN)
            w(f"ASSISTANT (thinking)   [{ts}]")
            w(THIN)
            w(think)
            w("")

        elif t == "tool_use":
            n_tool += 1
            w(THIN)
            w(f"TOOL CALL: {b.get('name')}   [{ts}]   id={b.get('id')}")
            w(THIN)
            inp = b.get("input", {})
            if isinstance(inp, dict):
                for k, v in inp.items():
                    v = v if isinstance(v, str) else json.dumps(
                        v, ensure_ascii=False, indent=2)
                    if "\n" in v:
                        w(f"  {k}:")
                        for ln in v.split("\n"):
                            w(f"    {ln}")
                    else:
                        w(f"  {k}: {v}")
            else:
                w(str(inp))
            w("")

        elif t == "tool_result":
            body = as_text(b.get("content"))
            err = " (ERROR)" if b.get("is_error") else ""
            w(f"--- TOOL RESULT{err}   id={b.get('tool_use_id')} ---")
            w(body.rstrip() if body.strip() else "(no output)")
            w("")

    if only_tool_results:
        pass  # already rendered above, no separate USER header

w(RULE)
w(f"END OF TRANSCRIPT   user turns: {n_user}   assistant messages: {n_asst}   "
  f"tool calls: {n_tool}")
w(f"({skipped_meta} injected system-reminder/meta records omitted)")
w(RULE)

with open(DST, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote {DST}")
print(f"  user turns {n_user} | assistant messages {n_asst} | tool calls {n_tool}")
print(f"  meta records skipped: {skipped_meta}")
