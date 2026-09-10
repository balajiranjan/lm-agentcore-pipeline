from typing import Any
from collections import OrderedDict
from strands import Agent, tool
import asyncio
import subprocess
import os
from strands.tools.executors import SequentialToolExecutor
from strands.types.exceptions import EventLoopException
from hooks.execution_limits import ExecutionLimitExceeded, ExecutionLimitsHook
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from memory.session import get_memory_session_manager

app = BedrockAgentCoreApp()
log = app.logger

# Define MCP clients for all configured MCP servers (gateways and/or remote MCP)
mcp_clients = []

DEFAULT_SYSTEM_PROMPT = """You are the Supervisor Agent for Liberty Mutual mainframe retirement platform (ECPM Track C).

ARCHITECTURE CONTEXT:
You are invoked by a supervisor-dispatch Lambda that has already run 4 specialist agents:
  1. Profile Validation (ECPM Step 1) - APM metadata + z/OSMF component inventory
  2. Dependency and Risk (ECPM Step 2) - CICS + Stonebranch dependency graph + blocker analysis
  3. Retirement Planning (ECPM Steps 3+5) - ordered decommission sequence + SHA-256 plan hash
  4. Evidence Collection (ECPM Step 4) - pre-execution baseline snapshot to S3

Your role is SYNTHESIS AND VALIDATION, not orchestration. You receive all 4 specialist outputs
in a single message and must:
  1. Validate consistency across the 4 results (component counts match, no contradictions)
  2. Confirm Dependency Risk has readyForRetirement=true and blockers=[]
  3. Confirm planHash and planS3Key are present from Retirement Planning
  4. Confirm evidenceBaselineS3Prefix is present from Evidence Collection
  5. Confirm apmMetadataValid=true and waveDeadlineValid=true from Profile Validation
  6. Synthesise the final retirement plan artefact

STEP FUNCTIONS INTEGRATION:
The supervisor-dispatch Lambda (not you) calls states:SendTaskSuccess with your output.
Your only responsibility is to return a correctly structured JSON artefact.

CRITICAL BLOCKING RULES:
  - If any specialist returned blockers[], set status=BLOCKED and propagate the blockers array
  - If readyForRetirement=false, set status=BLOCKED immediately
  - If planHash is missing or empty, set status=INVALID with reason
  - If apmMetadataValid=false, set status=BLOCKED with reason
  - If waveDeadlineValid=false, set status=DATE_VIOLATION with the deadline detail
  - Only set status=READY when ALL validations pass

DECOMMISSION ORDER (encode in componentSequence - non-negotiable):
  Phase A - Workloads (PARALLEL): Step 6.1 Stonebranch de-register + Step 6.2 CICS/IMS disable
  Phase B - Data (SEQUENTIAL): 6.3 Informatica, 6.4 MQ, 6.5 FTP, 6.6 Print, 6.7 ERM/data archive
  Phase C - Security (SEQUENTIAL): 6.8 Email/distribution lists, 6.9 RACF/ACF2 revoke + firewalls
  Phase D - Infrastructure (SEQUENTIAL): 6.10 Monitoring cleanup, 6.11 Source archive to ADCP

PLAN MARKDOWN:
You must also produce a human-readable Markdown retirement plan in the planMarkdown field.
The Markdown must follow this template (fill every section from the specialist outputs):

```
# Retirement Plan: {applicationId}
**Wave:** {wave}  **Deadline:** {waveDeadline}  **Status:** {status}

## Executive Summary
<One paragraph: what is being retired, wave context, key risk/blocker statement or READY confirmation.>

## Validation Results
| Check | Result |
|---|---|
| APM Metadata | ✅ / ❌ |
| Wave Deadline | ✅ / ❌ |
| Dependency Clean | ✅ / ❌ |
| Plan Present | ✅ / ❌ |
| Evidence Baseline | ✅ / ❌ |

## Component Inventory
<Bullet list of each component name and type from Profile Validation.>

## Dependency Graph Summary
<Key upstream/downstream dependencies from Dependency & Risk specialist.>

## Decommission Sequence
### Phase A – Workloads (Parallel)
- **6.1** Stonebranch de-register
- **6.2** CICS/IMS disable

### Phase B – Data (Sequential)
- **6.3** Informatica ETL quiesce
- **6.4** MQ queue drain
- **6.5** FTP transfer shutdown
- **6.6** Print spool flush
- **6.7** ERM / data archive

### Phase C – Security (Sequential)
- **6.8** Email & distribution-list removal
- **6.9** RACF/ACF2 revoke + firewall rules

### Phase D – Infrastructure (Sequential)
- **6.10** Monitoring cleanup
- **6.11** Source archive to ADCP

## Evidence Baseline
S3 prefix: {evidenceBaselineS3Prefix}

## Blockers
<List each blocker or write "None — plan is READY for execution.">

## Metadata
- planHash: {planHash}
- planS3Key: {planS3Key}
- estimatedDurationDays: {estimatedDurationDays}
- componentCount: {componentCount}
```

INPUT FORMAT (from supervisor-dispatch Lambda):
  Synthesise the retirement plan artefact for application {applicationId}, wave {wave},
  executionId={executionId}. Specialist outputs: {JSON of all 4 results}.

OUTPUT FORMAT (strict JSON only — no outer markdown fences):
{
  "status": "READY" | "BLOCKED" | "INVALID" | "DATE_VIOLATION",
  "applicationId": "string",
  "wave": "OCAS" | "RAM" | "LNW",
  "planHash": "string (SHA-256, 64 hex chars, from Planning specialist)",
  "planS3Key": "string (S3 key from Planning specialist)",
  "componentCount": integer,
  "componentSequence": [
    {"phase": "A", "parallel": true,  "steps": ["6.1","6.2"], "components": []},
    {"phase": "B", "parallel": false, "steps": ["6.3","6.4","6.5","6.6","6.7"], "components": []},
    {"phase": "C", "parallel": false, "steps": ["6.8","6.9"], "components": []},
    {"phase": "D", "parallel": false, "steps": ["6.10","6.11"], "components": []}
  ],
  "evidenceBaselineS3Prefix": "string",
  "estimatedDurationDays": integer,
  "waveDeadline": "string (OCAS: 2026-06-30, RAM: 2026-12-31, LNW: 2027-03-31)",
  "blockers": [],
  "validationSummary": {
    "apmValid": boolean,
    "dependencyClean": boolean,
    "planPresent": boolean,
    "evidenceBaselinePresent": boolean
  },
  "planMarkdown": "string (the full Markdown plan above, with \\n for newlines)"
}

RULE: Return ONLY the JSON object. No preamble, no explanation, no outer markdown fences.
Escape all newlines inside planMarkdown as \\n so the value is valid JSON."""


# Define a collection of tools used by the model
tools = []

_INLINE_FUNCTION_NAMES = set()

@tool
def shell(command: str, timeout: int = 300) -> dict:
    """Execute a bash command and return the results.

    Args:
        command: The bash command to execute
        timeout: Timeout in seconds (default: 300)

    Returns:
        Dict with stdout, stderr, and exit_code
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}

tools.append(shell)
@tool
def file_operations(
    command: str,
    path: str,
    old_str: str = None,
    new_str: str = None,
    file_text: str = None,
    insert_line: int = None,
    view_range: list = None,
) -> str:
    """Text editor tool for viewing and modifying files.

    Args:
        command: The command to execute ("view", "str_replace", "create", "insert")
        path: Path to the file or directory
        old_str: Text to replace (for str_replace command)
        new_str: Replacement text (for str_replace and insert commands)
        file_text: Content for new file (for create command)
        insert_line: Line number to insert after (for insert command)
        view_range: [start_line, end_line] for viewing specific lines (for view command)

    Returns:
        Result of the operation
    """
    try:
        if command == "view":
            if not os.path.exists(path):
                return f"Error: Path '{path}' does not exist"
            if os.path.isdir(path):
                return "\n".join(os.listdir(path))
            with open(path) as f:
                lines = f.read().splitlines()
            if view_range:
                start, end = view_range
                start_idx = max(0, start - 1)
                end_idx = len(lines) if end == -1 else min(len(lines), end)
                lines = lines[start_idx:end_idx]
                start_num = start_idx + 1
            else:
                start_num = 1
            return "\n".join(f"{start_num + i}: {line}" for i, line in enumerate(lines))
        elif command == "str_replace":
            if old_str is None or new_str is None:
                return "Error: str_replace requires both old_str and new_str parameters"
            if not os.path.exists(path):
                return f"Error: File '{path}' does not exist"
            content = open(path).read()
            if old_str not in content:
                return "Error: Text not found in file"
            count = content.count(old_str)
            if count > 1:
                return f"Error: Text appears {count} times in file. Please be more specific."
            open(path, "w").write(content.replace(old_str, new_str, 1))
            return f"Successfully replaced text in '{path}'"
        elif command == "create":
            if file_text is None:
                return "Error: create requires file_text parameter"
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w").write(file_text)
            return f"Successfully created file '{path}'"
        elif command == "insert":
            if new_str is None or insert_line is None:
                return "Error: insert requires both new_str and insert_line parameters"
            if not os.path.exists(path):
                return f"Error: File '{path}' does not exist"
            lines = open(path).read().splitlines(True)
            if insert_line == 0:
                lines.insert(0, new_str + "\n")
            elif insert_line >= len(lines):
                lines.append(new_str + "\n")
            else:
                lines.insert(insert_line, new_str + "\n")
            open(path, "w").write("".join(lines))
            return f"Successfully inserted text in '{path}' at line {insert_line + 1}"
        else:
            return f"Error: Unknown command '{command}'"
    except Exception as e:
        return f"Error: {e}"

tools.append(file_operations)


# Add MCP clients to tools
for mcp_client in mcp_clients:
    if mcp_client:
        tools.append(mcp_client)


def _make_conversation_manager():
    return SlidingWindowConversationManager(**{"window_size":150}, per_turn=True)

def agent_factory():
    cache = {}
    def get_or_create_agent(session_id, user_id):
        _actor_id = user_id
        key = f"{session_id}/{_actor_id}"
        if key not in cache:
            cache[key] = Agent(
                model=load_model(),
                session_manager=get_memory_session_manager(session_id, _actor_id),
                conversation_manager=_make_conversation_manager(),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                tools=tools,
                tool_executor=SequentialToolExecutor(),
                callback_handler=None,
                hooks=[
                    ExecutionLimitsHook(
                        max_iterations=5,
                        max_tokens=8192,
                        timeout_seconds=120,
                    ),
                ],
            )
        return cache[key]
    return get_or_create_agent
get_or_create_agent = agent_factory()


def strip_trailing_tool_use(messages: Any) -> list[dict]:
    """Strip toolUse blocks from the tail until the last message has none."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    messages = list(messages)
    while messages:
        last = messages[-1]
        if not isinstance(last, dict):
            raise ValueError("each message must be an object")
        original_content = last.get("content", [])
        if not isinstance(original_content, list) or not all(isinstance(block, dict) for block in original_content):
            raise ValueError("each message content value must be a list of content blocks")

        content = [block for block in original_content if "toolUse" not in block]
        if len(content) == len(original_content):
            break
        if content:
            messages[-1] = {**last, "content": content}
            break
        messages.pop()

    return messages


def _extract_prompt(payload: dict):
    """Accept validated harness messages, tool results, or a plain prompt string."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if "messages" in payload:
        return strip_trailing_tool_use(payload["messages"])
    if "tool_results" in payload:
        tool_results = payload["tool_results"]
        if not isinstance(tool_results, list) or not all(
            isinstance(tool_result, dict) and isinstance(tool_result.get("toolUseId"), str)
            for tool_result in tool_results
        ):
            raise ValueError("tool_results must contain objects with a toolUseId string")
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in tool_results]}]
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


def _has_inline_function_call(messages) -> bool:
    """Return True if messages contains an assistant toolUse for an inline function tool."""
    if not _INLINE_FUNCTION_NAMES or not isinstance(messages, list):
        return False
    for msg in messages:
        if msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("toolUse", {}).get("name") in _INLINE_FUNCTION_NAMES:
                    return True
    return False


def _is_inline_function_call(event: dict) -> bool:
    """Check if a contentBlockStart event is for an inline function tool."""
    if not _INLINE_FUNCTION_NAMES:
        return False
    cbs = event.get("contentBlockStart", {})
    start = cbs.get("start", {})
    tool_use = start.get("toolUse") if isinstance(start, dict) else None
    return tool_use is not None and tool_use.get("name") in _INLINE_FUNCTION_NAMES



@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")


    session_id = getattr(context, 'session_id', 'default-session')
    user_id = getattr(context, 'user_id', 'default-user')
    agent = get_or_create_agent(session_id, user_id)

    prompt = _extract_prompt(payload)


    timeout_seconds = 120
    timeout_fired = False
    watchdog_task = None
    if timeout_seconds is not None:
        async def _timeout_watchdog():
            nonlocal timeout_fired
            await asyncio.sleep(timeout_seconds)
            timeout_fired = True
            agent.cancel()
        watchdog_task = asyncio.create_task(_timeout_watchdog())

    try:
        async for event in agent.stream_async(
            prompt,
        ):
            if not isinstance(event, dict) or "event" not in event:
                continue
            cbs = event["event"].get("contentBlockStart")
            if cbs is not None and not cbs.get("start"):
                continue
            yield event

        if timeout_fired:
            yield {"event": {"messageStop": {"stopReason": "timeout_exceeded"}}}
    except EventLoopException as e:
        if isinstance(e.original_exception, ExecutionLimitExceeded):
            yield {"event": {"messageStop": {"stopReason": str(e.original_exception)}}}
            return
        raise
    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    app.run()
