#!/usr/bin/env python3
"""
HDD Disassembly Script — Pure state-machine, no LLM.

Flow:
  1. Subscribe to vision, wait for detections
  2. Hold the HDD chassis / top lid (whichever is visible)
  3. Unscrew all screws with spatial memory (skip already-visited zones)
  4. Pickup PCB main → verify removal in vision → retry if still detected
  5. Re-hold chassis → flip
  6. Done
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
import json, time, threading, math, copy

from disassembly_skills.unscrew_skill import UnscrewSkill
from disassembly_skills.object_hold_skill import ObjectHoldSkill
from disassembly_skills.object_flip_skill import ObjectFlipSkill
from disassembly_skills.object_pickup_skill import PickupSkill


class HDDDisassemblyScript(Node):
    def __init__(self):
        super().__init__('hdd_disassembly_script')

        # ── Skill nodes (added to executor separately) ──────────────────
        self.unscrew_skill = UnscrewSkill()
        self.hold_skill    = ObjectHoldSkill()
        self.flip_skill    = ObjectFlipSkill()
        self.pickup_skill  = PickupSkill()

        # ── Vision state ────────────────────────────────────────────────
        self.vision_lock     = threading.Lock()
        self.detected_objects = []
        self.vision_received = threading.Event()
        self.create_subscription(
            String, '/vision/agent_state', self._vision_cb, 10
        )

        # ── Spatial memory (screw exclusion zones — same as master agent) ─
        self.cleared_zones    = []
        self.EXCLUSION_RADIUS = 0.010  # 10 mm

        # ── Vision reset publisher ──────────────────────────────────────
        self.vision_reset_pub = self.create_publisher(
            String, '/vision/reset_tracker', 10
        )

        self.get_logger().info("🚀 HDD Disassembly Script: Ready.")

    # ==================================================================
    # Vision callback — filters detections using spatial memory
    # ==================================================================
    def _vision_cb(self, msg):
        try:
            data = json.loads(msg.data.strip().strip("'").strip('"'))
            raw = data.get("global_view", {}).get("objects", [])
            filtered = []
            for o in raw:
                xyz = o.get("xyz")
                if xyz and len(xyz) >= 3:
                    skip = False
                    for cz in self.cleared_zones:
                        dist = math.hypot(
                            xyz[0] - cz[0],
                            xyz[1] - cz[1],
                            xyz[2] - cz[2],
                        )
                        if dist < self.EXCLUSION_RADIUS:
                            skip = True
                            break
                    if skip:
                        continue
                filtered.append(o)
            with self.vision_lock:
                self.detected_objects = filtered
            self.vision_received.set()
        except Exception:
            pass

    def get_objects(self):
        """Return a snapshot of current detections."""
        with self.vision_lock:
            return copy.deepcopy(self.detected_objects)

    def wait_for_vision(self, timeout=10.0):
        """Block until at least one vision frame arrives."""
        self.vision_received.clear()
        return self.vision_received.wait(timeout=timeout)

    # ==================================================================
    # Finders — fuzzy keyword matching on label
    # ==================================================================
    def find_by_keywords(self, objects, keywords):
        """Return all objects whose label contains ANY keyword (case-insensitive)."""
        results = []
        for o in objects:
            label = o.get("label", "").lower()
            if any(kw.lower() in label for kw in keywords):
                results.append(o)
        return results

    def find_chassis(self, objects):
        return self.find_by_keywords(objects, ["hdd", "chassis", "lid"])

    def find_screws(self, objects):
        return self.find_by_keywords(objects, ["screw"])

    def find_pcb(self, objects):
        return self.find_by_keywords(objects, ["pcb"])

    # ==================================================================
    # STEP 1 — Hold the HDD chassis / top lid
    # ==================================================================
    def step_hold_chassis(self):
        """Find and hold the HDD chassis or top lid."""
        print("\n" + "=" * 60)
        print("STEP: HOLD CHASSIS / LID")
        print("=" * 60)

        for attempt in range(3):
            self.wait_for_vision()
            objects = self.get_objects()
            chassis_list = self.find_chassis(objects)

            if not chassis_list:
                print(f"  ⚠️ No HDD chassis/lid detected (attempt {attempt+1}/3). Waiting...")
                time.sleep(2.0)
                continue

            target = chassis_list[0]
            tid, tlabel = target["id"], target["label"]
            print(f"  🎯 Found: {tlabel} (ID: {tid})")
            print(f"  🗜️ Calling hold_skill.execute_hold({tid}, '{tlabel}')...")

            success = self.hold_skill.execute_hold(
                part_id=tid, target_label=tlabel, interactive=False
            )
            if success:
                print("  ✅ Hold successful.")
                return True
            else:
                print("  ❌ Hold failed. Retrying...")
                time.sleep(1.0)

        print("  ❌ Could not hold chassis after 3 attempts.")
        return False

    # ==================================================================
    # STEP 2 — Unscrew all screws (with spatial memory)
    # ==================================================================
    def step_unscrew_all(self):
        """Find and unscrew all visible screws, skipping already-cleared zones."""
        print("\n" + "=" * 60)
        print("STEP: UNSCREW ALL SCREWS (with spatial memory)")
        print("=" * 60)

        while True:
            time.sleep(1.0)  # Let vision update
            self.wait_for_vision()
            objects = self.get_objects()
            screws = self.find_screws(objects)

            if not screws:
                print("  ✅ No more screws detected. Moving on.")
                return True

            target = screws[0]
            tid, tlabel = target["id"], target["label"]
            target_xyz = target.get("xyz")
            print(f"\n  🔩 Unscrewing: {tlabel} (ID: {tid})")

            self.unscrew_skill.execute_unscrew_command(
                target_id=tid, target_label=tlabel, interactive=False
            )

            # Add to spatial memory regardless of outcome
            if target_xyz:
                self.cleared_zones.append(target_xyz)
                print(
                    f"  🧠 [MEMORY] Exclusion zone added at "
                    f"{[f'{v:.3f}' for v in target_xyz]}"
                )

    # ==================================================================
    # STEP 3 — Pickup PCB main + verify removal in vision
    # ==================================================================
    def step_pickup_pcb(self):
        """Pick up the PCB main and verify it's no longer visible."""
        print("\n" + "=" * 60)
        print("STEP: PICKUP PCB MAIN")
        print("=" * 60)

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            self.wait_for_vision()
            objects = self.get_objects()
            pcbs = self.find_pcb(objects)

            if not pcbs:
                print("  ✅ No PCB detected — already removed or not present.")
                return True

            target = pcbs[0]
            tid, tlabel = target["id"], target["label"]
            print(f"  📦 Picking up: {tlabel} (ID: {tid}) — attempt {attempt+1}/{MAX_RETRIES}")

            self.pickup_skill.execute_pickup(
                target_id=tid, target_label=tlabel
            )

            # ── Verify removal in vision ────────────────────────────────
            print("  🔍 Verifying removal in vision...")
            time.sleep(2.0)
            self.wait_for_vision()
            objects_after = self.get_objects()
            pcbs_after = self.find_pcb(objects_after)

            if not pcbs_after:
                print("  ✅ PCB removed successfully — not visible in vision.")
                return True
            else:
                print(
                    f"  ⚠️ PCB still detected ({len(pcbs_after)} found). "
                    f"Retrying pickup..."
                )
                time.sleep(1.0)

        print(f"  ❌ Could not remove PCB after {MAX_RETRIES} attempts.")
        return False

    # ==================================================================
    # STEP 4 — Re-hold chassis and flip
    # ==================================================================
    def step_hold_and_flip(self):
        """Re-hold the chassis (pickup released grip) then flip."""
        print("\n" + "=" * 60)
        print("STEP: HOLD + FLIP")
        print("=" * 60)

        # Re-hold (the pickup step releases the grip)
        print("  🗜️ Re-holding chassis before flip...")
        if not self.step_hold_chassis():
            print("  ❌ Re-hold failed. Cannot flip.")
            return False

        # Clear screw memory for the new side
        self.cleared_zones.clear()
        print("  🧠 [MEMORY] Cleared exclusion zones for new side.")

        # Reset vision tracker for fresh detections
        self.vision_reset_pub.publish(String(data='reset'))
        time.sleep(1.0)

        print("  🔄 Flipping...")
        success = self.flip_skill.execute_flip(interactive=False)
        if success:
            print("  ✅ Flip complete!")
        else:
            print("  ❌ Flip failed.")
        return success

    # ==================================================================
    # Main sequence
    # ==================================================================
    def run(self):
        print("\n" + "#" * 60)
        print("#  HDD DISASSEMBLY SCRIPT — START")
        print("#" * 60)

        # ── Wait for first vision frame ─────────────────────────────────
        print("\n⏳ Waiting for vision data...")
        while not self.vision_received.wait(timeout=2.0):
            print("  Still waiting for /vision/agent_state...")

        objects = self.get_objects()
        print(f"👁️ Vision online: {len(objects)} objects detected.")
        for o in objects:
            print(f"   - [{o.get('id', '?')}] {o.get('label', '?')}")

        # Reset vision tracker
        self.vision_reset_pub.publish(String(data='reset'))
        time.sleep(1.0)

        # ── 1. Hold HDD chassis / lid ───────────────────────────────────
        if not self.step_hold_chassis():
            print("\n❌ ABORT: Failed to hold chassis.")
            return

        # ── 2. Unscrew all screws (spatial memory) ──────────────────────
        self.step_unscrew_all()

        # ── 3. Pickup PCB main + visual verification ────────────────────
        if not self.step_pickup_pcb():
            print("\n⚠️ PCB pickup failed, continuing to flip anyway...")

        # ── 4. Re-hold + flip ───────────────────────────────────────────
        self.step_hold_and_flip()

        print("\n" + "#" * 60)
        print("#  HDD DISASSEMBLY SCRIPT — COMPLETE")
        print("#" * 60)


# ======================================================================
# Entry point
# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = HDDDisassemblyScript()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(node.unscrew_skill)
    executor.add_node(node.hold_skill)
    executor.add_node(node.flip_skill)
    executor.add_node(node.pickup_skill)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(2.0)  # Let subscriptions connect

    try:
        node.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
