#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Script entrypoint - Python 3.10.6
# -----------------------------------------------------------------------------
import os
import asyncio
import inspect
import re
import ast
import time
from typing import List, Tuple, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from disassembly_skills.disassembly_skills.unscrew_skill import UnscrewSkill
from disassembly_skills.disassembly_skills.object_hold_skill import ObjectHoldSkill
# -----------------------------------------------------------------------------
# OpenAI (remote) — async client
# -----------------------------------------------------------------------------
from openai import AsyncOpenAI, OpenAIError

DEBUG_FULL_OUTPUT = False

# Load API key from local file
API_KEY_FILE = "src/api_key.txt"
try:
    with open(API_KEY_FILE, "r") as f:
        OPENAI_API_KEY = f.read().strip()
except FileNotFoundError:
    raise RuntimeError(f"Missing {API_KEY_FILE}. Please create it with your OpenAI API key.")

# Choose model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

# Create client with key from file
oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Tool Class + async tools
# -----------------------------------------------------------------------------
class Tool:
    def __init__(self, fn, description: str, args: Optional[Dict[str, str]] = None):
        self.fn = fn
        self.description = description
        self.args = args or {}

# -----------------------------------------------------------------------------
# Terminal color utilities
# -----------------------------------------------------------------------------
class TColor:
    RESET = "\033[0m"

    REASONING   = "\033[38;5;244m"   # soft gray
    PLAN        = "\033[38;5;39m"    # blue
    ACTION      = "\033[38;5;214m"   # orange
    OBSERVATION = "\033[38;5;82m"    # green
    FINAL       = "\033[38;5;201m"   # magenta
    ERROR       = "\033[38;5;196m"   # red

def print_stage(text: str):
    if text.startswith("Reasoning:"):
        color = TColor.REASONING
    elif text.startswith("Plan:"):
        color = TColor.PLAN
    elif text.startswith("Action:"):
        color = TColor.ACTION
    elif text.startswith("Observation:"):
        color = TColor.OBSERVATION
    elif text.startswith("Final Answer:"):
        color = TColor.FINAL
    else:
        color = TColor.ERROR

    print(color + text + TColor.RESET, flush=True)

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
async def dummy(dumvar=100):
    return "Tool called successfully"

async def unscrew(unscrew_id= None, unscrew_label = None, hold_id = None, hold_label = None):
    if None in (unscrew_id, unscrew_label, hold_id, hold_label):
        return "Action failed, missing required parameter"
    else:
        unscrew = UnscrewSkill()
        hold = ObjectHoldSkill()

        hold.execute_hold(part_id=hold_id, target_label=hold_label)
        unscrew.execute_unscrew_command(target_id = unscrew_id, target_label=unscrew_label)

    return "Tool called successfully"

async def lift_and_drop(object_id= None, object_name = None):
    if object_id == None or object_name == None:
        return "Action failed, missing object_id or object_name parameter"
    else:
        pass
    
    return "Tool called successfully"

tools: Dict[str, Tool] = {
    "unscrew": Tool(
        unscrew,
        "Performs an unscrewing action. Requires the ID and label of the part to be unscrewed as well as the ID and label for another part to be held to keep the entire object secure during unscrewing",
        args={"unscrew_id": "int: ID of the target screw",
              "unscrew_label": "str: label of the target screw",
              "hold_id": "int: ID of the holding target",
              "hold_label": "str: label of the holding target"}
    )
}

def format_tool_list() -> str:
    lines = []
    for name, tool in tools.items():
        if tool.args:
            args_str = ", ".join(f"{arg}: {desc}" for arg, desc in tool.args.items())
            lines.append(f"- {name}({args_str}): {tool.description}")
        else:
            lines.append(f"- {name}(): {tool.description}")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------
FORBIDDEN_HEADERS = ("Plan:", "Action:", "Observation:", "Final Answer:")

def build_think_prompt(user_input: str, history: List[str]) -> str:
    tool_list = format_tool_list()
    guide = "\n".join([
        "You are a part of a ReAct agent. Your role is to produce a reasoning stage to reason about the problem and guide the following Plan, Action and Observation stages that will be handled by the rest of the agent.",
        "This stage is a TRANSPARENT scratchpad that will be shown to the user.",
        "",
        "Rules:",
        "- Output MUST start with: Reasoning:",
        "- You may write multiple lines after 'Reasoning:'.",
        "- Do NOT output Plan:, Action:, Observation:, or Final Answer: in this stage.",
        "- Do NOT refer to your role in the output. only produce relevant reasoning that can be used by the other parts of the agent"
        "- Be concrete: if applicable, summarise the previous plan> action> observation within the context of the user query and create a short generation to support the next planning stage",
        "",
        "Current Scene:",])
    return "\n".join([
        guide,
        "Tools available:",
        tool_list,
        "",
        f"User: {user_input}",
        *history,
        ""
    ])

def build_plan_prompt(user_input: str, history: List[str]) -> str:
    tool_list = format_tool_list()
    guide = "\n".join([
        "You are a ReAct agent controlling a robot arm.",
        "Output exactly ONE line.",
        "It MUST start with: Plan:  (or you may output Final Answer: if the task is complete).",
        "",
        "Rules:",
        "- Output exactly ONE line only.",
        "- Do NOT output Action:, Observation:, or Reasoning: here.",
        "- If uncertain, make a cautious Plan that leads to an Action next.",
        "",
    ])
    return "\n".join([
        guide,
        "Tools available:",
        tool_list,
        "",
        f"User: {user_input}",
        *history,
        ""
    ])

def build_action_prompt(user_input: str, history: List[str]) -> str:
    tool_list = format_tool_list()
    guide = "\n".join([
        "You are a ReAct agent controlling a robot arm.",
        "Output exactly ONE line.",
        "It MUST start with: Action:",
        "",
        "Rules:",
        "- Output exactly ONE line only.",
        "- Do NOT output Plan:, Observation:, Reasoning:, or Final Answer: here.",
        "- Use ONLY the tool names/signatures exactly as listed.",
        "- Do not invent extra keyword arguments.",
        "",
    ])
    return "\n".join([
        guide,
        "Tools available:",
        tool_list,
        "",
        f"User: {user_input}",
        *history,
        ""
    ])

# -----------------------------------------------------------------------------
# Remote LLM calls
# -----------------------------------------------------------------------------
FIRST_LINE_RE = re.compile(r"([^\r\n]*)")

async def call_llm_remote_first_line(prompt: str, max_tokens: int = 200, timeout_s: int = 180) -> str:
    try:
        resp = await asyncio.wait_for(
            oai_client.responses.create(
                model=OPENAI_MODEL,
                reasoning={"effort": "none"},
                input=prompt,
                max_output_tokens=max_tokens
            ),
            timeout=timeout_s
        )

        text = getattr(resp, "output_text", None)
        if not text:
            parts = []
            for item in resp.output or []:
                if getattr(item, "type", None) == "message":
                    for c in (item.content or []):
                        if getattr(c, "type", None) == "output_text":
                            parts.append(getattr(c, "text", ""))
            text = "".join(parts).strip()

        if not text:
            return ""

        if DEBUG_FULL_OUTPUT:
            print("=== FULL MODEL OUTPUT START ===")
            print(text)
            print("=== FULL MODEL OUTPUT END ===")

        m = FIRST_LINE_RE.match(text.strip())
        return m.group(1).strip() if m else text.splitlines()[0].strip()

    except (asyncio.TimeoutError, OpenAIError) as e:
        return f"Plan: (API error: {type(e).__name__}) proceed cautiously."

async def call_llm_remote_full(prompt: str, max_tokens: int = 800, timeout_s: int = 180) -> str:
    try:
        resp = await asyncio.wait_for(
            oai_client.responses.create(
                model=OPENAI_MODEL,
                reasoning={"effort": "none"},
                input=prompt,
                max_output_tokens=max_tokens
            ),
            timeout=timeout_s
        )

        text = getattr(resp, "output_text", None)
        if not text:
            parts = []
            for item in resp.output or []:
                if getattr(item, "type", None) == "message":
                    for c in (item.content or []):
                        if getattr(c, "type", None) == "output_text":
                            parts.append(getattr(c, "text", ""))
            text = "".join(parts).strip()

        if not text:
            return ""

        if DEBUG_FULL_OUTPUT:
            print("=== FULL MODEL OUTPUT START ===")
            print(text)
            print("=== FULL MODEL OUTPUT END ===")

        return text.strip()

    except (asyncio.TimeoutError, OpenAIError) as e:
        return f"Reasoning: (API error: {type(e).__name__})"

# -----------------------------------------------------------------------------
# Reasoning sanitizer (prevents header leakage into history)
# -----------------------------------------------------------------------------
def sanitize_reasoning(txt: str) -> str:
    txt = (txt or "").strip()
    if not txt.startswith("Reasoning:"):
        txt = "Reasoning:\n" + txt

    lines = txt.splitlines()
    out = [lines[0]]  # keep "Reasoning:" line

    for line in lines[1:]:
        s = line.strip()
        if any(s.startswith(h) for h in FORBIDDEN_HEADERS):
            # drop leaked stage headers entirely
            continue
        out.append(line)

    return "\n".join(out).strip()

# -----------------------------------------------------------------------------
# Action parser — supports positional + keyword args, strings/numbers/bools/None
# -----------------------------------------------------------------------------
ACTION_RE = re.compile(r"^Action:\s*([\w_]+)\s*\((.*)\)\s*$")

def _const_or_name(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        return -node.operand.value if isinstance(node.op, ast.USub) else +node.operand.value
    raise ValueError("Unsupported argument expression")

def parse_action(response: str) -> Tuple[Optional[str], List[Any], Dict[str, Any]]:
    m = ACTION_RE.match(response.strip())
    if not m:
        return None, [], {}
    name, args_str = m.group(1), (m.group(2) or "").strip()
    if args_str == "":
        return name, [], {}

    try:
        node = ast.parse(f"f({args_str})", mode="eval")
        call = node.body
        if not isinstance(call, ast.Call):
            return name, [], {}
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        for a in call.args:
            try:
                args.append(_const_or_name(a))
            except ValueError:
                pass
        for kw in call.keywords:
            key = kw.arg
            if key is None:
                continue
            try:
                kwargs[key] = _const_or_name(kw.value)
            except ValueError:
                pass
        return name, args, kwargs
    except SyntaxError:
        return name, [], {}

# -----------------------------------------------------------------------------
# Agent state
# -----------------------------------------------------------------------------
class AgentState(dict):
    input: str
    history: List[str]
    phase: str  # "plan" or "action"

# -----------------------------------------------------------------------------
# LangGraph nodes: think -> plan -> action -> act
# -----------------------------------------------------------------------------
async def think(state: AgentState) -> AgentState:
    txt = await call_llm_remote_full(build_think_prompt(state["input"], state["history"]), max_tokens=800)
    txt = sanitize_reasoning(txt)

    state["history"].append(txt)
    print_stage(txt)
    return state

async def plan_node(state: AgentState) -> AgentState:
    line = await call_llm_remote_first_line(build_plan_prompt(state["input"], state["history"]), max_tokens=120)

    # Allow Final Answer here, otherwise require Plan:
    if line.startswith("Final Answer:"):
        state["history"].append(line)
        print_stage(line)
        return state

    if not line.startswith("Plan:"):
        line = "Plan: Prepare next simple step"

    state["history"].append(line)
    print_stage(line)
    state["phase"] = "action"
    return state

async def action_node(state: AgentState) -> AgentState:
    # If already ended, skip
    if state["history"] and state["history"][-1].startswith("Final Answer:"):
        return state

    line = await call_llm_remote_first_line(build_action_prompt(state["input"], state["history"]), max_tokens=120)

    if not line.startswith("Action:"):
        # Safe fallback: go back to planning next loop if it fails to comply
        line = "Action: dummy()"

    state["history"].append(line)
    print_stage(line)
    return state

async def act(state: AgentState) -> AgentState:
    if not state["history"]:
        return state

    last = state["history"][-1]

    if last.startswith("Final Answer:"):
        return state

    if last.startswith("Action:"):
        name, args, kwargs = parse_action(last)
        tool = tools.get(name or "")

        if not tool:
            obs = f"Observation: Unknown tool '{name}'. Available: {', '.join(tools.keys())}"
        else:
            fn = tool.fn
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                obs = f"Observation: {result}"
            except Exception as e:
                obs = f"Observation: Tool '{name}' errored: {type(e).__name__}: {e}"

        state["history"].append(obs)
        print_stage(obs)

    return state

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
async def run_agent(user_query: str = "") -> List[str]:
    graph = StateGraph(AgentState)

    graph.add_node("think", think)
    graph.add_node("plan", plan_node)
    graph.add_node("action", action_node)
    graph.add_node("act", act)

    graph.set_entry_point("think")
    graph.add_edge("think", "plan")
    graph.add_edge("plan", "action")
    graph.add_edge("action", "act")

    def _branch(s: AgentState) -> str:
        return "end" if any(line.startswith("Final Answer:") for line in s["history"]) else "continue"

    graph.add_conditional_edges("act", _branch, {"continue": "think", "end": END})

    app = graph.compile()

    state: AgentState = {"input": user_query, "history": [], "phase": "plan"}
    print("PROMPT:", state["input"], flush=True)

    async for _ in app.astream(state, config={"recursion_limit": 400}):
        pass

    return state["history"]

# -----------------------------------------------------------------------------
# Script entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_agent(user_query="Can you call the dummy tool twice, one at a time?"))
