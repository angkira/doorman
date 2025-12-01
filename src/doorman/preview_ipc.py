#!/usr/bin/env python3
"""
Live camera preview using frame streaming from doorman daemon

This is a GUI client that receives camera frames from the daemon
and overlays face detection/recognition results from the debug stream.

The daemon owns the camera exclusively and broadcasts frames in preview mode.
The preview client simply displays frames with detection overlays.

Usage:
    # Start daemon in preview mode
    ~/bin/doormand --user --preview

    # Start preview client
    doorman preview
"""

import cv2
import json
import logging
import os
import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# Auto-detect debug socket (user service vs system service)
def get_debug_socket_path() -> str:
    """Get debug socket path - prefer user socket if it exists"""
    user_runtime_dir = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    user_socket = f"{user_runtime_dir}/doorman-debug.sock"
    system_socket = "/run/doorman-debug.sock"

    if os.path.exists(user_socket):
        return user_socket
    elif os.path.exists(system_socket):
        return system_socket
    else:
        return user_socket  # Default to user

def get_frame_socket_path() -> str:
    """Get frame socket path - prefer user socket if it exists"""
    user_runtime_dir = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    user_socket = f"{user_runtime_dir}/doorman-frames.sock"
    system_socket = "/run/doorman-frames.sock"

    if os.path.exists(user_socket):
        return user_socket
    elif os.path.exists(system_socket):
        return system_socket
    else:
        return user_socket  # Default to user

DEBUG_SOCKET_PATH = get_debug_socket_path()
FRAME_SOCKET_PATH = get_frame_socket_path()


class DebugStreamClient:
    """Client that subscribes to daemon's debug stream."""

    def __init__(self, socket_path: str = DEBUG_SOCKET_PATH):
        self.socket_path = socket_path
        self.sock = None
        self.latest_detection = None
        self.running = False
        self.thread = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to debug stream socket."""
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            self.connected = True
            print(f"✓ Connected to daemon debug stream at {self.socket_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to debug stream: {e}")
            self.connected = False
            return False

    def start(self) -> bool:
        """Start listening to debug stream in background thread."""
        if not self.connected and not self.connect():
            return False

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        """Background thread that reads from debug stream."""
        buffer = b""

        while self.running:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    print("Debug stream disconnected")
                    break

                buffer += chunk

                # Process complete messages (newline-delimited JSON)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        try:
                            message = json.loads(line.decode())
                            self.latest_detection = message
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse debug message: {e}")

            except Exception as e:
                if self.running:
                    print(f"Error reading debug stream: {e}")
                break

        self.connected = False

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the latest detection result from daemon."""
        return self.latest_detection

    def stop(self):
        """Stop listening and close connection."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


class FrameStreamClient:
    """Client that subscribes to daemon's frame stream."""

    def __init__(self, socket_path: str = FRAME_SOCKET_PATH):
        self.socket_path = socket_path
        self.sock = None
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.connected = False
        self.frame_count = 0

    def connect(self) -> bool:
        """Connect to frame stream socket."""
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            self.connected = True
            print(f"✓ Connected to daemon frame stream at {self.socket_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to frame stream: {e}")
            self.connected = False
            return False

    def start(self) -> bool:
        """Start listening to frame stream in background thread."""
        if not self.connected and not self.connect():
            return False

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        """Background thread that reads frames from stream."""
        while self.running:
            try:
                # Read frame size (4 bytes, big-endian u32)
                size_bytes = self._recv_exactly(4)
                if not size_bytes:
                    print("Frame stream disconnected")
                    break

                frame_size = struct.unpack('>I', size_bytes)[0]

                # Read JPEG data
                jpeg_data = self._recv_exactly(frame_size)
                if not jpeg_data:
                    print("Failed to receive complete frame")
                    break

                # Decode JPEG to numpy array
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.latest_frame = frame
                    self.frame_count += 1

            except Exception as e:
                if self.running:
                    print(f"Error reading frame stream: {e}")
                break

        self.connected = False

    def _recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(min(4096, n - len(data)))
            if not chunk:
                return None
            data += chunk
        return data

    def get_latest_frame(self):
        """Get the latest frame from daemon."""
        return self.latest_frame

    def stop(self):
        """Stop listening and close connection."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


class FacePreview:
    """Live camera preview showing daemon detection results."""

    def __init__(self, camera_index: int = 0):
        """Initialize the preview."""
        self.camera_index = camera_index  # Not used anymore, daemon owns camera
        self.debug_client = DebugStreamClient()
        self.frame_client = FrameStreamClient()

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0.0

        # Colors (BGR format)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.WHITE = (255, 255, 255)
        self.ORANGE = (0, 165, 255)


    def draw_text(
        self,
        frame,
        text: str,
        position: tuple,
        color: tuple,
        font_scale: float = 0.6,
    ) -> None:
        """Draw text on frame with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        cv2.rectangle(
            frame,
            (x - 5, y - text_h - baseline - 5),
            (x + text_w + 5, y + baseline),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA
        )

    def update_fps(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_update >= 0.5:
            elapsed = current_time - self.last_fps_update
            frames_since_last = self.frame_count
            self.fps = frames_since_last / elapsed if elapsed > 0 else 0
            self.frame_count = 0  # Reset counter for next period
            self.last_fps_update = current_time

    def process_frame(self, frame) -> None:
        """Process a frame and add overlays from detection data."""
        self.update_fps()

        # Get latest detection data from debug stream
        detection_msg = self.debug_client.get_latest()

        if detection_msg:
            detection = detection_msg.get("detection", {})
            bbox = detection.get("bbox")

            if bbox:
                # Face detected
                # Daemon sends bbox as (x, y, width, height) in pixels
                x, y, w, h = bbox
                confidence = detection.get("confidence", 0.0)
                recognized_user = detection.get("recognized_user")
                similarity = detection.get("similarity")
                
                # Debug logging
                frame_h, frame_w = frame.shape[:2]
                logger.debug(f"Frame size: {frame_w}x{frame_h}")
                logger.debug(f"BBox from daemon (x,y,w,h): {x}, {y}, {w}, {h}")
                logger.debug(f"Drawing rectangle: ({x},{y}) to ({x+w},{y+h})")

                # Choose color based on recognition
                if recognized_user:
                    color = self.GREEN
                    status = f"✓ Recognized: {recognized_user} ({similarity:.2f})"
                    status_color = self.GREEN
                else:
                    color = self.RED
                    status = f"✓ Face Detected (conf: {confidence:.2f})"
                    status_color = self.RED

                # Draw bounding box
                # bbox is (x, y, width, height), so bottom-right corner is (x+w, y+h)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            else:
                # No face detected
                status = "👤 No Face Detected"
                status_color = self.ORANGE

            # Draw status
            self.draw_text(frame, status, (10, 30), status_color)

            # Draw processing info
            processing_time = detection_msg.get("processing_time_ms", 0)
            info_text = f"Preview: {self.fps:.1f} FPS | Processing: {processing_time}ms"
            self.draw_text(frame, info_text, (10, 60), self.WHITE)
        else:
            # Not connected to daemon yet
            status = "⚠️  Waiting for daemon..."
            self.draw_text(frame, status, (10, 30), self.YELLOW)
            info_text = f"Preview: {self.fps:.1f} FPS"
            self.draw_text(frame, info_text, (10, 60), self.WHITE)

        # Draw help text
        help_text = "Press 'q' or ESC to quit"
        self.draw_text(frame, help_text, (10, frame.shape[0] - 20), self.YELLOW, 0.5)

    def run(self) -> None:
        """Run the preview loop using frame streaming from daemon."""
        print("\n" + "=" * 60)
        print("Preview started. Press 'q' or ESC to quit.")
        print("=" * 60 + "\n")

        # Connect to frame stream
        if not self.frame_client.start():
            print("\n❌ Failed to connect to daemon frame stream")
            print("\nTroubleshooting:")
            print("   - Ensure daemon is running in preview mode: ~/bin/doormand --user --preview")
            print("   - Check daemon logs: journalctl --user -u doormand -f")
            return

        # Connect to debug stream
        if not self.debug_client.start():
            print("\n⚠️  Failed to connect to debug stream (detection data)")
            print("Continuing anyway, will show frames without detection overlay...\n")

        try:
            print("✅ Connected to daemon streams, displaying frames...\n")

            while True:
                # Get latest frame from daemon
                frame = self.frame_client.get_latest_frame()

                if frame is None:
                    # No frame yet, wait a bit
                    time.sleep(0.01)
                    continue

                # Process frame and add overlays
                self.process_frame(frame)
                cv2.imshow("Doorman Face Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    print("\n👋 User requested quit")
                    break

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            # Cleanup
            self.frame_client.stop()
            self.debug_client.stop()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore errors on cleanup

            # Print statistics
            duration = time.time() - self.start_time
            print(f"\n📊 Session Statistics:")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Frames received: {self.frame_client.frame_count}")
            print(f"   Average FPS: {self.fps:.1f}")
            print("\n✨ Preview stopped\n")

    def run_console(self, debug: bool = False) -> None:
        """Run console-only preview (text output, no GUI window)."""
        print("\n" + "=" * 60)
        print("Console Preview Mode - Press Ctrl+C to quit")
        print("=" * 60 + "\n")

        # Connect to debug stream
        if not self.debug_client.start():
            print("\n❌ Failed to connect to daemon")
            print("\nTroubleshooting:")
            print("   - Ensure daemon is running: ~/bin/doormand --user --preview")
            print("   - Check daemon logs: journalctl --user -u doormand -f")
            return

        print("✅ Connected to daemon\n")

        try:
            last_update = time.time()
            while True:
                # Get latest detection data from debug stream
                detection_msg = self.debug_client.get_latest()

                # Update every ~2 seconds
                current_time = time.time()
                if current_time - last_update < 2.0:
                    time.sleep(0.1)
                    continue

                last_update = current_time

                if detection_msg:
                    detection = detection_msg.get("detection", {})
                    bbox = detection.get("bbox")
                    processing_time = detection_msg.get("processing_time_ms", 0)
                    fps = 1000.0 / processing_time if processing_time > 0 else 0

                    timestamp = time.strftime("%H:%M:%S")

                    if bbox:
                        # Face detected
                        x, y, w, h = bbox
                        confidence = detection.get("confidence", 0.0)
                        recognized_user = detection.get("recognized_user")
                        similarity = detection.get("similarity")

                        if recognized_user:
                            print(f"[{timestamp}] ✅ Recognized: {recognized_user} (similarity: {similarity:.2f}) | FPS: {fps:.1f}")
                        else:
                            print(f"[{timestamp}] 👤 Face Detected (conf: {confidence:.2f}) - bbox: ({int(x)}, {int(y)}, {int(w)}x{int(h)}) | FPS: {fps:.1f}")

                        if debug:
                            print(f"           Processing time: {processing_time}ms")
                            print(f"           Bbox: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
                    else:
                        # No face detected
                        print(f"[{timestamp}] ⚠️  No Face Detected | FPS: {fps:.1f}")
                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
        finally:
            # Cleanup
            self.debug_client.stop()

            # Print statistics
            duration = time.time() - self.start_time
            print(f"\n📊 Session Statistics:")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Average FPS: {self.fps:.1f}")
            print("\n✨ Console preview stopped\n")

def main():
    """Main entry point."""
    print("Starting Doorman preview (frame streaming mode)...")
    preview = FacePreview()
    preview.run()


if __name__ == "__main__":
    main()
