#!/usr/bin/env python3

import os
import asyncio
import re
import threading
import json
import time
from typing import List, Dict, Any

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# OpenAI
from openai import AsyncOpenAI

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
    VISION      = "\033[38;5;226m"   # yellow

def print_stage(text: str):
    prefix_map = {
        "Reasoning:": TColor.REASONING,
        "Plan:": TColor.PLAN,
        "Action:": TColor.ACTION,
        "Observation:": TColor.OBSERVATION,
        "Final Answer:": TColor.FINAL,
        "Vision:": TColor.VISION
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
        
        # 1. Threading & Sync
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.loop_thread.start()
        
        # Global Storage for Vision Data
        self.vision_lock = threading.Lock()
        self.detected_objects = [] 
        
        # --- NEW: Snapshot Control ---
        self.has_printed_startup_vision = False 
        
        # 2. Resolve API Key
        try:
            api_key_path = "/home/adip/workspaces/disassembly_ws/src/disassembly_skills/config/api_key.txt"
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            self.get_logger().error(f"Could not load API key: {e}")
            raise RuntimeError("API Key Missing.")

        # 3. Setup OpenAI
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.oai_client = AsyncOpenAI(api_key=api_key)

        # 4. Instantiate Sub-Skill Nodes
        self.unscrew_skill = UnscrewSkill()
        self.hold_skill = ObjectHoldSkill()

        # 5. Define Tools
        self.tools = {
            "unscrew": {
                "fn": self.unscrew_wrapper,
                "args": ["unscrew_id", "unscrew_label", "hold_id", "hold_label"]
            },
            "hold_object": {
                "fn": self.hold_wrapper,
                "args": ["part_id", "label"]
            }
        }

        # 6. ROS Interfaces
        self.goal_sub = self.create_subscription(String, '/agent_goal', self.goal_callback, 10)
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        self.get_logger().info("🤖 Master Agent Ready. Waiting for vision snapshot...")

    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -------------------------------------------------------------------------
    # Vision Data Handling (Startup Print Only)
    # -------------------------------------------------------------------------
    def vision_callback(self, msg):
        """Updates global list and prints the tracking table exactly once at startup."""
        try:
            clean_json = msg.data.strip("'")
            data = json.loads(clean_json)
            raw_objects = data.get("global_view", {}).get("objects", [])
            
            temp_list = []
            for obj in raw_objects:
                temp_list.append({
                    "id": obj.get("id"),
                    "label": obj.get("label"),
                    "xyz": obj.get("xyz")
                })

            with self.vision_lock:
                self.detected_objects = temp_list

            # --- PRINT ONLY ONCE ON PROGRAM START ---
            if not self.has_printed_startup_vision and len(self.detected_objects) > 0:
                print("\n" + "="*60)
                print_stage(f"Vision: Startup Snapshot ({len(self.detected_objects)} parts detected).")
                print("-" * 60)
                
                for item in self.detected_objects:
                    xyz = item['xyz']
                    # Format XYZ to 2 decimal places
                    fmt_xyz = [round(val, 2) for val in xyz] if xyz else "None"
                    print(f"   ID {item['id']:<2} | {item['label']:<25} | XYZ: {fmt_xyz}")
                
                print("="*60 + "\n")
                self.has_printed_startup_vision = True

        except Exception as e:
            self.get_logger().error(f"Vision Error: {e}", throttle_duration_sec=10.0)

    # --- Tool Wrappers ---
    async def hold_wrapper(self, part_id: int, label: str):
        self.get_logger().info(f"🛠️ Executing Hold Skill on {label}...")
        success = self.hold_skill.execute_hold(part_id=int(part_id), target_label=label, interactive=False)
        return "SUCCESS: Object secured" if success else "FAILURE: Could not hold"

    async def unscrew_wrapper(self, unscrew_id: int, unscrew_label: str, hold_id: int, hold_label: str):
        self.get_logger().info(f"🛠️ Executing Unscrew Skill on {unscrew_label}...")
        success = self.unscrew_skill.execute_unscrew_command(target_id=int(unscrew_id), target_label=unscrew_label, interactive=False)
        return "SUCCESS: Screw removed" if success else "FAILURE: Unscrew failed"

    # --- ReAct Agent Loop ---
    def get_prompt_header(self):
        with self.vision_lock:
            vision_context = json.dumps(self.detected_objects)
        return (f"You are a robotic disassembly agent.\n"
                f"CURRENT VISIBLE OBJECTS: {vision_context}\n\n"
                "Format: Reasoning: <thought>\nPlan: <next_step>\nAction: tool_name(args)\n"
                "Observation: <result>\nFinal Answer: <summary>")

    async def call_llm(self, prompt, max_tokens=400):
        try:
            res = await self.oai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.get_prompt_header()},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0
            )
            return res.choices[0].message.content.strip()
        except Exception as e: return f"Error: {e}"

    async def agent_loop(self, user_query):
        self.get_logger().info(f"🧠 Reasoning about: {user_query}")
        state = {"input": user_query, "history": []}
        for _ in range(10): 
            prompt = f"Goal: {state['input']}\nHistory:\n" + "\n".join(state["history"]) + "\nNext Step:"
            llm_out = await self.call_llm(prompt)
            for line in llm_out.splitlines():
                if line.startswith(("Reasoning:", "Plan:", "Action:", "Final Answer:")):
                    print_stage(line); state["history"].append(line)
            last_entry = state["history"][-1]
            if "Final Answer:" in last_entry: break
            if "Action:" in last_entry:
                match = re.search(r"Action:\s*(\w+)\((.*)\)", last_entry)
                if match:
                    t_name, t_args_raw = match.groups()
                    t_args = [a.strip().strip("'").strip('"') for a in t_args_raw.split(",") if a.strip()]
                    if t_name in self.tools:
                        obs = await self.tools[t_name]["fn"](*t_args)
                        obs_str = f"Observation: {obs}"
                        print_stage(obs_str); state["history"].append(obs_str)

    def goal_callback(self, msg):
        self.get_logger().info(f"Received Goal: {msg.data}")
        asyncio.run_coroutine_threadsafe(self.agent_loop(msg.data), self.loop)

def main(args=None):
    rclpy.init(args=args)
    master_agent = MasterAgentNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(master_agent)
    executor.add_node(master_agent.unscrew_skill)
    executor.add_node(master_agent.hold_skill)
    try:
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        master_agent.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()