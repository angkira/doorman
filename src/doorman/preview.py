#!/usr/bin/env python3
"""
Live camera preview synchronized with doorman daemon detection

This is a GUI client that displays the camera feed and overlays
face detection/recognition results from the daemon's debug stream.

The daemon does all ML processing, this just displays results.

Usage:
    doorman preview
"""

import cv2
import json
import logging
import os
import socket
import threading
import time
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

DEBUG_SOCKET_PATH = get_debug_socket_path()


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
            logger.info(f"✓ Connected to daemon debug stream at {self.socket_path}")
            print(f"✓ Connected to daemon debug stream at {self.socket_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to debug stream: {e}")
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
                            # Debug: print first detection to verify format
                            if hasattr(self, '_first_detection_logged') == False:
                                self._first_detection_logged = True
                                print(f"[DEBUG] First detection received: {message}")
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


class FacePreview:
    """Live camera preview showing daemon detection results."""

    def __init__(self, camera_index: int = 0):
        """Initialize the preview."""
        self.camera_index = camera_index
        self.camera = None
        self.debug_client = DebugStreamClient()

        # IPC socket path for requesting detection
        self.ipc_socket_path = None
        self.last_detection_request = 0
        self.detection_interval = 0.1  # Request detection every 100ms

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

    def connect_ipc(self) -> bool:
        """Get IPC socket path (connection is created per-request)."""
        try:
            user_runtime_dir = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            self.ipc_socket_path = f"{user_runtime_dir}/doorman.sock"

            # Test connection
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.ipc_socket_path)
            sock.close()
            return True
        except Exception as e:
            print(f"⚠️  Failed to connect to daemon IPC: {e}")
            return False

    def request_detection(self, debug: bool = False) -> Optional[Dict[str, Any]]:
        """Request face detection from daemon."""
        if not self.ipc_socket_path:
            if debug:
                print("[DEBUG] No IPC socket path set")
            return None

        current_time = time.time()
        if current_time - self.last_detection_request < self.detection_interval:
            if debug:
                print(f"[DEBUG] Rate limited: {current_time - self.last_detection_request:.3f}s < {self.detection_interval}s")
            return None  # Too soon

        self.last_detection_request = current_time
        if debug:
            print(f"[DEBUG] Sending detection request to {self.ipc_socket_path}")

        # Create new connection for each request (daemon closes after each response)
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(3.0)  # 3 second timeout (camera capture + ML inference can take time)
            sock.connect(self.ipc_socket_path)
            if debug:
                print("[DEBUG] Socket connected")

            # Send DetectAndRecognize request
            request = {"type": "detect_and_recognize"}
            sock.sendall((json.dumps(request) + "\n").encode())
            if debug:
                print("[DEBUG] Request sent")

            # Read response
            response_data = b""
            while b"\n" not in response_data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk

            if debug:
                print(f"[DEBUG] Response received: {len(response_data)} bytes")

            if response_data:
                response = json.loads(response_data.decode().strip())
                if debug:
                    print(f"[DEBUG] Parsed response: {response}")
                if response.get("status") == "success" and response.get("data"):
                    result = response["data"].get("result")
                    if debug:
                        print(f"[DEBUG] Returning result: {result}")
                    return result
                else:
                    if debug:
                        print(f"[DEBUG] Response not success or no data")
        except socket.timeout:
            if debug:
                print("[DEBUG] Socket timeout")
            pass  # Timeout is ok, just skip this frame
        except Exception as e:
            print(f"⚠️  Detection request failed: {e}")
        finally:
            if sock:
                sock.close()

        if debug:
            print("[DEBUG] Returning None")
        return None

    def open_camera(self) -> bool:
        """Open the camera device using GStreamer with PipeWire integration."""
        print(f"📷 Opening camera {self.camera_index} via GStreamer/PipeWire...")

        # GStreamer pipeline for PipeWire camera access
        gst_pipeline = (
            f"pipewiresrc ! "
            f"videoconvert ! "
            f"videoscale ! "
            f"video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! "
            f"appsink drop=true max-buffers=2"
        )

        print(f"🔧 Using GStreamer pipeline: {gst_pipeline}")
        self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.camera.isOpened():
            print(f"⚠️  Failed to open camera via GStreamer, trying V4L2 fallback...")
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                print(f"❌ Failed to open camera {self.camera_index}")
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Using V4L2 fallback")

        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))

        print(f"✅ Camera opened: {width}x{height} @ {fps}fps")
        return True

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
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time

    def process_frame(self, frame, debug: bool = False) -> None:
        """Process a frame and add overlays."""
        self.update_fps()

        # Get latest detection from debug stream (cached)
        debug_msg = self.debug_client.get_latest()

        if debug_msg:
            # Extract detection info from debug stream message
            detection = debug_msg.get("detection", {})
            bbox = detection.get("bbox")
            frame_size = detection.get("frame_size")
            
            if bbox and frame_size:
                # Face detected - scale bbox from daemon frame size to preview frame size
                daemon_width, daemon_height = frame_size
                preview_height, preview_width = frame.shape[:2]
                
                x, y, w, h = bbox
                
                # Debug: log bbox scaling
                if debug:
                    logger.info(f"Daemon frame: {daemon_width}x{daemon_height} | Preview frame: {preview_width}x{preview_height}")
                    logger.info(f"Original bbox: x={x}, y={y}, w={w}, h={h}")
                
                # Scale bbox coordinates
                x_scaled = int(x * preview_width / daemon_width)
                y_scaled = int(y * preview_height / daemon_height)
                w_scaled = int(w * preview_width / daemon_width)
                h_scaled = int(h * preview_height / daemon_height)
                
                if debug:
                    logger.info(f"Scaled bbox: x={x_scaled}, y={y_scaled}, w={w_scaled}, h={h_scaled}")
                
                confidence = detection.get("confidence", 0.0)
                recognized_user = detection.get("recognized_user")
                similarity = detection.get("similarity")

                # Choose color based on recognition
                if recognized_user:
                    color = self.GREEN
                    status = f"✓ Recognized: {recognized_user} ({similarity:.2f})"
                    status_color = self.GREEN
                else:
                    color = self.RED
                    status = f"✓ Face Detected (conf: {confidence:.2f})"
                    status_color = self.RED

                # Draw bounding box with scaled coordinates
                cv2.rectangle(frame, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, 3)
            else:
                # No face detected
                status = "👤 No Face Detected"
                status_color = self.ORANGE

            # Draw status
            self.draw_text(frame, status, (10, 30), status_color)

            # Draw processing info
            info_text = f"Preview: {self.fps:.1f} FPS"
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

    def run(self, debug: bool = False) -> None:
        """Run the preview loop."""
        print("\n" + "=" * 60)
        print("Preview started. Press 'q' or ESC to quit.")
        print("=" * 60 + "\n")

        # Connect to daemon debug stream
        if not self.debug_client.start():
            print("\nTroubleshooting:")
            print("   - Ensure daemon is running with --preview flag")
            print("   - Or start user daemon: doormand --user --preview")
            print("\nContinuing anyway, will wait for daemon...")

        if not self.open_camera():
            print("\nTroubleshooting:")
            print("   - Check camera connection: v4l2-ctl --list-devices")
            print("   - Try different camera index: doorman preview --camera 1")
            return

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                self.process_frame(frame, debug=debug)
                cv2.imshow("Doorman Face Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    print("\n👋 User requested quit")
                    break

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            # Cleanup
            self.debug_client.stop()
            if self.camera:
                self.camera.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore errors on cleanup

            # Print statistics
            duration = time.time() - self.start_time
            print(f"\n📊 Session Statistics:")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Frames: {self.frame_count}")
            print(f"   Average FPS: {self.fps:.1f}")
            print("\n✨ Preview stopped\n")

    def run_console(self, debug: bool = False) -> None:
        """Run the preview in console mode (text output only, no GUI)."""
        import sys
        # Force unbuffered output for real-time display
        sys.stdout.reconfigure(line_buffering=True)

        print("\n" + "=" * 60)
        print("Console Preview Mode - Press Ctrl+C to quit")
        print("=" * 60 + "\n")

        # Connect to daemon IPC
        if not self.connect_ipc():
            print("\n❌ Troubleshooting:")
            print("   - Ensure daemon is running: doorman status")
            print("   - Or start user daemon: doormand --user")
            return

        print("✅ Connected to daemon\n")
        if debug:
            print(f"[DEBUG] IPC socket path: {self.ipc_socket_path}")
            print(f"[DEBUG] Detection interval: {self.detection_interval}s\n")

        try:
            last_status = None
            frame_count = 0
            start_time = time.time()
            request_count = 0

            while True:
                # Request detection from daemon
                request_count += 1
                if debug:
                    print(f"\n[DEBUG] Loop iteration {request_count}")
                detection = self.request_detection(debug=debug)

                if detection:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    # Create status string
                    bbox = detection.get("bbox")
                    if bbox:
                        x, y, w, h = bbox
                        confidence = detection.get("confidence", 0.0)
                        recognized_user = detection.get("recognized_user")
                        similarity = detection.get("similarity")

                        if recognized_user:
                            status = f"✅ Recognized: {recognized_user} (similarity: {similarity:.2f})"
                        else:
                            status = f"👤 Face Detected (conf: {confidence:.2f}) - bbox: ({x}, {y}, {w}x{h})"
                    else:
                        status = "⚠️  No Face Detected"

                    # Only print if status changed or every 10 frames
                    if status != last_status or frame_count % 10 == 0:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] {status} | FPS: {fps:.1f}")
                        last_status = status

                time.sleep(0.1)  # 10 Hz polling

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
        finally:
            duration = time.time() - start_time
            print(f"\n📊 Session Statistics:")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Frames processed: {frame_count}")
            if duration > 0:
                print(f"   Average FPS: {frame_count / duration:.1f}")
            print("\n✨ Console preview stopped\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Doorman camera preview with face detection visualization"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Console mode: text output only, no GUI window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()
    
    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Create and run preview
    preview = FacePreview(camera_index=args.camera)
    if args.console:
        preview.run_console(debug=args.debug)
    else:
        preview.run(debug=args.debug)


if __name__ == "__main__":
    main()
