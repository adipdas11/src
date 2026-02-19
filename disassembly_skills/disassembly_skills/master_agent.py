#!/usr/bin/env python3

import os
import asyncio
import inspect
import re
import ast
import threading
from typing import List, Tuple, Optional, Dict, Any

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

# LangGraph & OpenAI
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI, OpenAIError

# Local Skill Imports
from disassembly_skills.unscrew_skill import UnscrewSkill
from disassembly_skills.object_hold_skill import ObjectHoldSkill

# -----------------------------------------------------------------------------
# Terminal Color Utilities
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
    prefix_map = {
        "Reasoning:": TColor.REASONING,
        "Plan:": TColor.PLAN,
        "Action:": TColor.ACTION,
        "Observation:": TColor.OBSERVATION,
        "Final Answer:": TColor.FINAL
    }
    color = TColor.ERROR
    for prefix, c in prefix_map.items():
        if text.startswith(prefix):
            color = c
            break
    print(color + text + TColor.RESET, flush=True)

# -----------------------------------------------------------------------------
# Master Agent Node
# -----------------------------------------------------------------------------
class MasterAgentNode(Node):
    def __init__(self):
        super().__init__('master_agent')
        
        # 1. Threading Setup: Create a dedicated thread for Asyncio LLM calls
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.loop_thread.start()
        
        # 2. Resolve API Key Path
        try:
            # Using absolute path as requested in your logs
            api_key_path = "/home/adip/workspaces/disassembly_ws/src/disassembly_skills/config/api_key.txt"
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            self.get_logger().error(f"Could not load API key from {api_key_path}: {e}")
            raise RuntimeError("API Key Missing.")

        # 3. Setup OpenAI Client
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.oai_client = AsyncOpenAI(api_key=api_key)

        # 4. Instantiate Sub-Skill Nodes
        self.unscrew_skill = UnscrewSkill()
        self.hold_skill = ObjectHoldSkill()

        # 5. Define Agent Tools
        self.tools = {
            "unscrew": {
                "fn": self.unscrew_wrapper,
                "desc": "Unscrews a part. Requires hold_id, unscrew_id, and labels.",
                "args": ["unscrew_id", "unscrew_label", "hold_id", "hold_label"]
            },
            "hold_object": {
                "fn": self.hold_wrapper,
                "desc": "Secures an object using the tactile hold skill.",
                "args": ["part_id", "label"]
            }
        }

        # 6. ROS Interfaces
        self.goal_sub = self.create_subscription(String, '/agent_goal', self.goal_callback, 10)
        self.get_logger().info("🤖 Master Agent Ready. Send goals to /agent_goal")

    def _run_async_loop(self):
        """Internal worker for the dedicated asyncio thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # --- Tool Wrappers ---
    async def hold_wrapper(self, part_id: int, label: str):
        self.get_logger().info(f"🛠️ Executing Hold Skill on {label}...")
        # Skill nodes run synchronously within this wrapper
        success = self.hold_skill.execute_hold(part_id=int(part_id), target_label=label, interactive=False)
        return "SUCCESS: Object secured" if success else "FAILURE: Could not hold object"

    async def unscrew_wrapper(self, unscrew_id: int, unscrew_label: str, hold_id: int, hold_label: str):
        self.get_logger().info(f"🛠️ Executing Unscrew Skill on {unscrew_label}...")
        success = self.unscrew_skill.execute_unscrew_command(target_id=int(unscrew_id), target_label=unscrew_label, interactive=False)
        return "SUCCESS: Screw removed and disposed" if success else "FAILURE: Unscrewing failed"

    # --- ReAct Agent Core ---
    def get_prompt_header(self):
        tool_desc = "\n".join([f"- {k}({', '.join(v['args'])}): {v['desc']}" for k, v in self.tools.items()])
        return (
            f"You are a robotic disassembly agent. Tools:\n{tool_desc}\n\n"
            "Format your response exactly as follows:\n"
            "Reasoning: <thought process>\n"
            "Plan: <short-term goal>\n"
            "Action: tool_name(args)\n"
            "Observation: <result will be provided>\n"
            "... repeat until done ...\n"
            "Final Answer: <summary of completion>"
        )

    async def call_llm(self, prompt, max_tokens=350):
        try:
            response = await self.oai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.get_prompt_header()},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    # --- Agent Execution Loop ---
    async def agent_loop(self, user_query):
        self.get_logger().info(f"🧠 Reasoning about: {user_query}")
        state = {"input": user_query, "history": []}
        
        for step in range(10): 
            prompt = f"User Goal: {state['input']}\nHistory:\n" + "\n".join(state["history"]) + "\nNext Step:"
            llm_out = await self.call_llm(prompt)
            
            # Process multi-line responses from LLM
            lines = llm_out.splitlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                
                if line.startswith(("Reasoning:", "Plan:", "Action:", "Final Answer:")):
                    print_stage(line)
                    state["history"].append(line)

            last_entry = state["history"][-1]
            if "Final Answer:" in last_entry:
                break
            
            if "Action:" in last_entry:
                # Regex to extract: tool_name(arg1, arg2...)
                match = re.search(r"Action:\s*(\w+)\((.*)\)", last_entry)
                if match:
                    t_name, t_args_raw = match.groups()
                    # Clean arguments: remove quotes and whitespace
                    t_args = [a.strip().strip("'").strip('"') for a in t_args_raw.split(",") if a.strip()]
                    
                    if t_name in self.tools:
                        obs = await self.tools[t_name]["fn"](*t_args)
                        obs_str = f"Observation: {obs}"
                        print_stage(obs_str)
                        state["history"].append(obs_str)
                    else:
                        error_msg = f"Observation: Tool {t_name} not found."
                        print_stage(error_msg)
                        state["history"].append(error_msg)

    def goal_callback(self, msg):
        self.get_logger().info(f"Received Topic Goal: {msg.data}")
        # Run the agent_loop in the dedicated asyncio thread to avoid thread errors
        asyncio.run_coroutine_threadsafe(self.agent_loop(msg.data), self.loop)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    
    # Create the master agent
    master_agent = MasterAgentNode()
    
    # Use MultiThreadedExecutor so the skills can process TF/Vision while the Agent waits for LLM
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(master_agent)
    executor.add_node(master_agent.unscrew_skill)
    executor.add_node(master_agent.hold_skill)
    
    

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        master_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()