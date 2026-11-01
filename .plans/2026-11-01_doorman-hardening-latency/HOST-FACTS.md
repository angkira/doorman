# HOST-FACTS (read-only collection, 2026-11-01T20:02:28+01:00)

## 1. Camera

### `v4l2-ctl --list-devices`
```
UGREEN Camera 4K: UGREEN Camera (<usb-bus>):
	/dev/video0
	/dev/video1
	/dev/media0
```

### `v4l2-ctl -d /dev/video0 --list-formats-ext`
```
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 3840x2160
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1920x1080
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 4000x3000
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 2560x1440
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 2592x1520
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 2048x1536
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x960
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1024x768
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 480x320
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x1024
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1920x1080
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x1024
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 2560x1440
			Interval: Discrete 0.333s (3.000 fps)
		Size: Discrete 2592x1944
			Interval: Discrete 0.333s (3.000 fps)
		Size: Discrete 1600x1200
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 480x320
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 1024x720
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 1280x960
			Interval: Discrete 0.100s (10.000 fps)
```

### `v4l2-ctl -d /dev/video0 --all`
```
Driver Info:
	Driver name      : uvcvideo
	Card type        : UGREEN Camera 4K: UGREEN Camera
	Bus info         : <usb-bus>
	Driver version   : 6.17.13
	Capabilities     : 0x84a00001
		Video Capture
		Metadata Capture
		Streaming
		Extended Pix Format
		Device Capabilities
	Device Caps      : 0x04200001
		Video Capture
		Streaming
		Extended Pix Format
Media Driver Info:
	Driver name      : uvcvideo
	Model            : UGREEN Camera 4K: UGREEN Camera
	Serial           : <serial>
	Bus info         : <usb-bus>
	Media version    : 6.17.13
	Hardware revision: 0x00000138 (312)
	Driver version   : 6.17.13
Interface Info:
	ID               : 0x03000002
	Type             : V4L Video
Entity Info:
	ID               : 0x00000001 (1)
	Name             : UGREEN Camera 4K: UGREEN Camera
	Function         : V4L2 I/O
	Flags            : default
	Pad 0x01000007   : 0: Sink
	  Link 0x02000010: from remote pad 0x100000a of entity 'Processing 3' (Video Pixel Formatter): Data, Enabled, Immutable
Priority: 2
Video input : 0 (Camera 1: ok)
Format Video Capture:
	Width/Height      : 1024/720
	Pixel Format      : 'YUYV' (YUYV 4:2:2)
	Field             : None
	Bytes per Line    : 2048
	Size Image        : 1474560
	Colorspace        : sRGB
	Transfer Function : Rec. 709
	YCbCr/HSV Encoding: ITU-R 601
	Quantization      : Default (maps to Limited Range)
	Flags             : 
Crop Capability Video Capture:
	Bounds      : Left 0, Top 0, Width 1024, Height 720
	Default     : Left 0, Top 0, Width 1024, Height 720
	Pixel Aspect: 1/1
Selection Video Capture: crop_default, Left 0, Top 0, Width 1024, Height 720, Flags: 
Selection Video Capture: crop_bounds, Left 0, Top 0, Width 1024, Height 720, Flags: 
Streaming Parameters Video Capture:
	Capabilities     : timeperframe
	Frames per second: 15.000 (15/1)
	Read buffers     : 0

User Controls

                     brightness 0x00980900 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                       contrast 0x00980901 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                     saturation 0x00980902 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                            hue 0x00980903 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                          gamma 0x00980910 (int)    : min=0 max=64 step=1 default=32 value=32 flags=has-min-max
                           gain 0x00980913 (int)    : min=0 max=15 step=1 default=0 value=0 flags=has-min-max
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=2 (60 Hz)
				0: Disabled
				1: 50 Hz
				2: 60 Hz
      white_balance_temperature 0x0098091a (int)    : min=1800 max=10000 step=1 default=5000 value=5000 flags=inactive, has-min-max
                      sharpness 0x0098091b (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
         backlight_compensation 0x0098091c (int)    : min=0 max=3 step=1 default=1 value=1 flags=has-min-max

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
				1: Manual Mode
				3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=100 value=100 flags=inactive, has-min-max
                   pan_absolute 0x009a0908 (int)    : min=-576000 max=576000 step=3600 default=0 value=0 flags=has-min-max
                  tilt_absolute 0x009a0909 (int)    : min=-576000 max=576000 step=3600 default=0 value=0 flags=has-min-max
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=512 value=490 flags=inactive, has-min-max
     focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
                  zoom_absolute 0x009a090d (int)    : min=0 max=100 step=1 default=0 value=0 flags=has-min-max
```

### `v4l2-ctl -d /dev/video0 --list-ctrls`
```

User Controls

                     brightness 0x00980900 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                       contrast 0x00980901 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                     saturation 0x00980902 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
                            hue 0x00980903 (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                          gamma 0x00980910 (int)    : min=0 max=64 step=1 default=32 value=32 flags=has-min-max
                           gain 0x00980913 (int)    : min=0 max=15 step=1 default=0 value=0 flags=has-min-max
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=2 (60 Hz)
      white_balance_temperature 0x0098091a (int)    : min=1800 max=10000 step=1 default=5000 value=5000 flags=inactive, has-min-max
                      sharpness 0x0098091b (int)    : min=1 max=64 step=1 default=32 value=32 flags=has-min-max
         backlight_compensation 0x0098091c (int)    : min=0 max=3 step=1 default=1 value=1 flags=has-min-max

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=100 value=100 flags=inactive, has-min-max
                   pan_absolute 0x009a0908 (int)    : min=-576000 max=576000 step=3600 default=0 value=0 flags=has-min-max
                  tilt_absolute 0x009a0909 (int)    : min=-576000 max=576000 step=3600 default=0 value=0 flags=has-min-max
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=512 value=490 flags=inactive, has-min-max
     focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
                  zoom_absolute 0x009a090d (int)    : min=0 max=100 step=1 default=0 value=0 flags=has-min-max
```

### `v4l2-ctl -d /dev/video1 --list-formats-ext`
```
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

```

### `v4l2-ctl -d /dev/video1 --all`
```
Driver Info:
	Driver name      : uvcvideo
	Card type        : UGREEN Camera 4K: UGREEN Camera
	Bus info         : <usb-bus>
	Driver version   : 6.17.13
	Capabilities     : 0x84a00001
		Video Capture
		Metadata Capture
		Streaming
		Extended Pix Format
		Device Capabilities
	Device Caps      : 0x04a00000
		Metadata Capture
		Streaming
		Extended Pix Format
Media Driver Info:
	Driver name      : uvcvideo
	Model            : UGREEN Camera 4K: UGREEN Camera
	Serial           : <serial>
	Bus info         : <usb-bus>
	Media version    : 6.17.13
	Hardware revision: 0x00000138 (312)
	Driver version   : 6.17.13
Interface Info:
	ID               : 0x03000005
	Type             : V4L Video
Entity Info:
	ID               : 0x00000004 (4)
	Name             : UGREEN Camera 4K: UGREEN Camera
	Function         : V4L2 I/O
Priority: 2
Format Metadata Capture:
	Sample Format   : 'UVCH' (UVC Payload Header Metadata)
	Buffer Size     : 10240
```

### `v4l2-ctl -d /dev/video1 --list-ctrls`
```
```

### `ffmpeg -version | head -1`
```
ffmpeg version 7.1.1-1ubuntu4.2 Copyright (c) 2000-2025 the FFmpeg developers
```

### `gst-inspect-1.0 --version`
```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
gst-inspect-1.0 version 1.26.6
GStreamer 1.26.6
https://launchpad.net/ubuntu/+source/gstreamer1.0
```

### `gst-inspect-1.0 v4l2src | head -15`
```
Factory Details:
  Rank                     primary (256)
  Long-name                Video (video4linux2) Source
  Klass                    Source/Video
  Description              Reads frames from a Video4Linux2 device
  Author                   (redacted third-party author)
  Documentation            https://gstreamer.freedesktop.org/documentation/video4linux2/v4l2src.html

Plugin Details:
  Name                     video4linux2
  Description              elements for Video 4 Linux
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvideo4linux2.so
  Version                  1.26.5
  License                  LGPL
  Source module            gst-plugins-good
```

## 2. nokhwa/V4L2 journal

### `journalctl -u doormand --since '2 days ago' --no-pager | grep -iE 'nokhwa|v4l2|gstreamer|ffmpeg|camera|backend' | tail -40`
```
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.467372Z  INFO doormand::camera::v4l2_backend: Initializing V4L2 camera backend (direct hardware access)
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.467405Z  INFO doormand::camera::v4l2_backend: Opening camera 0...
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.467603Z  WARN doormand::camera: V4L2/nokhwa camera failed: Failed to open camera, trying next backend
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.469201Z  INFO doormand::camera::ffmpeg_backend: Initializing ffmpeg webcam backend (device 0, 1280x720 @ 30fps)
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.470707Z  INFO doormand::camera: Using ffmpeg live webcam backend
Nov 01 11:03:13 <host> doormand[1859]: 2026-11-01T10:03:13.470745Z  INFO doormand::pipeline::camera_producer: Camera gated ON — opened in producer thread
Nov 01 11:03:32 <host> doormand[1859]: 2026-11-01T10:03:32.388473Z  INFO doormand::pipeline::camera_producer: Camera gated OFF (screen unlocked / no preview) — releasing device
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.976247Z  INFO doormand::camera::v4l2_backend: Initializing V4L2 camera backend (direct hardware access)
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.976286Z  INFO doormand::camera::v4l2_backend: Opening camera 0...
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.976720Z  WARN doormand::camera: V4L2/nokhwa camera failed: Failed to open camera, trying next backend
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.978699Z  INFO doormand::camera::ffmpeg_backend: Initializing ffmpeg webcam backend (device 0, 1280x720 @ 30fps)
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.980445Z  INFO doormand::camera: Using ffmpeg live webcam backend
Nov 01 13:02:17 <host> doormand[1859]: 2026-11-01T12:02:17.980500Z  INFO doormand::pipeline::camera_producer: Camera gated ON — opened in producer thread
Nov 01 13:02:27 <host> doormand[1859]: 2026-11-01T12:02:27.423634Z  INFO doormand::pipeline::camera_producer: Camera gated OFF (screen unlocked / no preview) — releasing device
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.335664Z  INFO doormand::camera::v4l2_backend: Initializing V4L2 camera backend (direct hardware access)
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.335681Z  INFO doormand::camera::v4l2_backend: Opening camera 0...
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.335803Z  WARN doormand::camera: V4L2/nokhwa camera failed: Failed to open camera, trying next backend
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.337413Z  INFO doormand::camera::ffmpeg_backend: Initializing ffmpeg webcam backend (device 0, 1280x720 @ 30fps)
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.338829Z  INFO doormand::camera: Using ffmpeg live webcam backend
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.338845Z  INFO doormand::pipeline::camera_producer: Camera gated ON — opened in producer thread
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.760156Z  INFO doormand::pipeline::camera_producer: Camera gated OFF (screen unlocked / no preview) — releasing device
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.871203Z  INFO doormand::camera::v4l2_backend: Initializing V4L2 camera backend (direct hardware access)
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.871232Z  INFO doormand::camera::v4l2_backend: Opening camera 0...
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.871370Z  WARN doormand::camera: V4L2/nokhwa camera failed: Failed to open camera, trying next backend
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.937488Z  INFO doormand::camera::ffmpeg_backend: Initializing ffmpeg webcam backend (device 0, 1280x720 @ 30fps)
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.938952Z  INFO doormand::camera: Using ffmpeg live webcam backend
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.938967Z  INFO doormand::pipeline::camera_producer: Camera gated ON — opened in producer thread
Nov 01 19:27:31 <host> doormand[1859]: 2026-11-01T18:27:31.124934Z  INFO doormand::pipeline::camera_producer: Camera gated OFF (screen unlocked / no preview) — releasing device
```

## 3. Runtime

### `lscpu | head -20`
```
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 7 8700G w/ Radeon 780M Graphics
CPU family:                              25
Model:                                   117
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               1
Stepping:                                2
Frequency boost:                         enabled
CPU(s) scaling MHz:                      85%
CPU max MHz:                             5380.3931
CPU min MHz:                             411.7650
BogoMIPS:                                8400.75
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d
```

### `ldconfig -p | grep -i onnxruntime`
```
	libonnxruntime_providers_shared.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libonnxruntime_providers_shared.so
	libonnxruntime_providers_dnnl.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libonnxruntime_providers_dnnl.so
	libonnxruntime.so.1.21 (libc6,x86-64) => /lib/x86_64-linux-gnu/libonnxruntime.so.1.21
```

### `dpkg -l | grep -iE "onnxruntime|doorman"`
```
ii  doorman                                                  0.1.0                                        amd64        Face authentication daemon and PAM module
ii  libonnxruntime-providers:amd64                           1.21.0+dfsg-2                                amd64        cross-platform inference and training ML accelerator (providers)
ii  libonnxruntime1.21:amd64                                 1.21.0+dfsg-2                                amd64        cross-platform inference and training ML accelerator (shared lib)
```

### `grep -i onnx /proc/$(systemctl show -p MainPID --value doormand)/maps | awk "{print \$6}" | sort -u`
```
grep: /proc/1859/maps: Permission denied
```

### `ls -l /var/lib/doorman 2>&1 | head -3; ls -ld /var/lib/doorman`
```
total 84
drwx------ 1 doorman doorman   984 Nov  1 13:12 dataset
-rw------- 1 doorman doorman 82378 Jun 24 17:42 embeddings.bin
drwxr-xr-x 1 doorman doorman 78 Jun 24 17:04 /var/lib/doorman
```

### `ls -l /usr/share/doorman /usr/lib/doorman /opt/doorman 2>&1 | head -30`
```
ls: cannot access '/opt/doorman': No such file or directory
/usr/lib/doorman:
total 279372
drwxr-xr-x 1 root root         8 Jun 24 14:41 adapters
-rw-r--r-- 1 root root  26768392 Jun 12 11:36 libonnxruntime.so
-rw-r--r-- 1 root root 259288040 Jun 12 11:36 libonnxruntime_providers_rocm.so
-rw-r--r-- 1 root root     15384 Jun 12 11:36 libonnxruntime_providers_shared.so

/usr/share/doorman:
total 4
-rw-r--r-- 1 root root 1671 Jun 24 17:41 doorman.toml.example
drwxr-xr-x 1 root root  186 Jun 24 17:42 models
```

## 4. Service

### `systemctl status doormand --no-pager | head -20`
```
● doormand.service - doorman Face Authentication Daemon (system)
     Loaded: loaded (/etc/systemd/system/doormand.service; enabled; preset: enabled)
    Drop-In: /etc/systemd/system/doormand.service.d
             └─10-drm.conf
     Active: active (running) since Sat 2026-10-17 18:20:06 CEST; 2 weeks 1 day ago
 Invocation: d303016fc6444b99a8f2d20e68cdb788
   Main PID: 1859 (doormand)
      Tasks: 42 (limit: 128)
     Memory: 463.4M (max: 2G, available: 1.5G, peak: 611.3M, swap: 353.3M, swap peak: 568.9M)
        CPU: 8min 31.478s
     CGroup: /system.slice/doormand.service
             └─1859 /usr/bin/doormand --config /etc/doorman/doorman.toml --device cpu

Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.625014Z  INFO doormand::pipeline::detection_pipeline: Broadcasting detection: bbox=(686, 421, 229, 196) = top_left + size, confidence=0.919
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.641430Z  INFO doormand::ipc: Authentication aggregate for <user>: 10 quality frames, cosine=0.7651 (threshold 0.40)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.705990Z  INFO doormand::ipc: Authentication successful for <user> (similarity: 0.7651)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.781518Z  INFO doormand::pipeline::detection_pipeline: Broadcasting detection: bbox=(685, 420, 232, 198) = top_left + size, confidence=0.917
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.784885Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.77, frame: 489)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.874386Z  INFO doormand::pipeline::detection_pipeline: Broadcasting detection: bbox=(691, 423, 225, 193) = top_left + size, confidence=0.915
Nov 01 19:27:31 <host> doormand[1859]: 2026-11-01T18:27:31.006655Z  INFO doormand::pipeline::detection_pipeline: Broadcasting detection: bbox=(692, 429, 225, 194) = top_left + size, confidence=0.913
```

### `systemctl cat doormand`
```
# /etc/systemd/system/doormand.service
# doorman face-authentication daemon — SYSTEM service.
# Generated by `doorman install` (EP: ncnn). Do not edit by hand; re-run the installer.
[Unit]
Description=doorman Face Authentication Daemon (system)
After=systemd-udev-settle.service local-fs.target
Wants=systemd-udev-settle.service
# CRASH CAP (BUG-3): after 5 failed starts within 60s systemd gives up and
# marks the unit `failed` instead of restarting forever, so a crashing EP
# cannot crash-loop and flood coredumps (the 4.2GB-core incident).
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=doorman
Group=doorman
SupplementaryGroups=video render
Environment=RUST_LOG=info
ExecStart=/usr/bin/doormand --config /etc/doorman/doorman.toml --device cpu
Restart=on-failure
RestartSec=2s
# CRASH CAP (BUG-3): disable core dumps entirely — a SEGV must not write a
# 250MB+ core; thousands of them flooded the disk with 4.2GB. LimitCORE=0
# guarantees a crash leaves NO core.
LimitCORE=0
StandardOutput=journal
StandardError=journal
SyslogIdentifier=doormand
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
RestrictAddressFamilies=AF_UNIX
StateDirectory=doorman
RuntimeDirectory=doorman
ReadWritePaths=/var/lib/doorman
DevicePolicy=closed
DeviceAllow=char-video4linux rw
# char-drm (DRM major 226) grants the /dev/dri/renderD*/card* nodes. A bare
# `DeviceAllow=/dev/dri rw` does NOT cover those char devices under
# DevicePolicy=closed, so the ncnn/Vulkan (radv) backend got
# "renderD128: Operation not permitted" and fell back to CPU. Needed for
# Vulkan (ncnn iGPU depth) and harmless for the other EPs.
DeviceAllow=char-drm rw
DeviceAllow=/dev/dri rw
DeviceAllow=/dev/kfd rw
MemoryMax=2G
TasksMax=128

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/doormand.service.d/10-drm.conf
[Service]
DeviceAllow=char-drm rw
```

### `systemd-analyze security doormand --no-pager | tail -25`
```
✗ CapabilityBoundingSet=~CAP_SYSLOG                           Service has access to kernel logging                                                                                         0.1
✗ DeviceAllow=                                                Service has a device ACL with some special devices: char-drm:rw /dev/kfd:rw /dev/dri:rw char-drm:rw char-video4linux:rw      0.1
✓ KeyringMode=                                                Service doesn't share key material with other services                                                                  
✓ Delegate=                                                   Service does not maintain its own delegated control group subtree                                                       
✗ SystemCallFilter=~@clock                                    Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@cpu-emulation                            Service does not filter system calls                                                                                         0.1
✗ SystemCallFilter=~@debug                                    Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@module                                   Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@mount                                    Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@obsolete                                 Service does not filter system calls                                                                                         0.1
✗ SystemCallFilter=~@privileged                               Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@raw-io                                   Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@reboot                                   Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@resources                                Service does not filter system calls                                                                                         0.2
✗ SystemCallFilter=~@swap                                     Service does not filter system calls                                                                                         0.2
✗ IPAddressDeny=                                              Service does not define an IP address allow list                                                                             0.2
✓ NotifyAccess=                                               Service child processes cannot alter service state                                                                      
✓ PrivateMounts=                                              Service cannot install system mounts                                                                                    
✓ RestrictAddressFamilies=~AF_PACKET                          Service cannot allocate packet sockets                                                                                  
✓ RestrictAddressFamilies=~AF_NETLINK                         Service cannot allocate netlink sockets                                                                                 
✓ RestrictAddressFamilies=~…                                  Service cannot allocate exotic sockets                                                                                  
✓ RestrictAddressFamilies=~AF_(INET|INET6)                    Service cannot allocate Internet sockets                                                                                
✗ UMask=                                                      Files created by service are world-readable by default                                                                       0.1

→ Overall exposure level for doormand.service: 7.6 EXPOSED 🙁
```

### `journalctl -u doormand --no-pager --since '2 days ago' | grep -iE 'denied|permission|seccomp|apparmor' | tail -20`
```
```

## 5. Config

### `cat /etc/doorman/doorman.toml`
```
# doorman daemon configuration (system service).
#
# Installed to /etc/doorman/doorman.toml by `make install` if no config exists.
# All keys are optional; omitted keys fall back to the documented defaults.

[daemon]
socket_path = "/run/doorman/doorman.sock"
data_dir    = "/var/lib/doorman"
log_level   = "info"
# Real deployment: boot LOCKED and clear the lock only on a recognized face.
# The background recognition loop processes frames while locked. PAM
# authentication runs an on-demand face check regardless of this flag.
start_locked = true
# Leave false for the system service; only the dev/preview build streams frames.
preview_mode = false
debug_mode   = false

[camera]
# V4L2 device index (/dev/video0 -> 0). The lean Linux build uses the
# camera-v4l2 or camera-gstreamer backend selected at compile time.
device_index = 0
width  = 1280
height = 720
fps    = 30

[ml]
backend    = "ort"
device     = "cpu"            # "rocm" / "cuda" only with the matching feature build
models_dir = "/var/lib/doorman/models"
cpu_threads = 0              # 0 = let ONNX Runtime pick

[authentication]
# Cosine-similarity match threshold for EdgeFace-S (see shared/src/lib.rs).
similarity_threshold = 0.4
auth_frames = 10
timeout_secs         = 3     # must stay <= the PAM hard timeout
# MiniFASNet liveness is a convenience deterrent; non-fatal when it fails.
liveness_enabled     = true
proactive_unlock = false  # Sway adapter owns face unlock

[recognition]
aggregation_window = 5
hysteresis_frames = 3
min_sharpness = 8.0
min_face_area_frac = 0.005

[dataset]
logging_enabled = true
max_attempt_records_per_day = 5000
keep_raw_frames = true
raw_frame_bundle_size = 10
max_capture_packs_per_day = 500
raw_frame_jpeg_quality = 92
raw_face_frame_bundle_size = 120
```

### `ls /etc/doorman | head -40`
```
doorman.toml
doorman.toml.bak
doorman.toml.bak-20260624-142113
doorman.toml.bak-20260624-142242
doorman.toml.bak-auth-latency-20260626-113422
doorman.toml.bak-auth-latency-20260626-123941
doorman.toml.bak-dataset-20260624-163744
doorman.toml.bak-dataset-20260624-170334
doorman.toml.bak-dataset-20260624-170343
doorman.toml.bak-dataset-20260624-171345
doorman.toml.bak-dataset-20260624-174211
doorman.toml.bak-dataset-20260626-113422
doorman.toml.bak-dataset-20260626-123941
doorman.toml.bak-disable-unlock-20260624-153549
doorman.toml.bak-recognition
doorman.toml.bak-sway-adapter-20260624-154446
doorman.toml.bak-sway-adapter-20260624-155831
doorman.toml.bak-sway-adapter-20260624-160957
doorman.toml.doorman-bak
doorman.toml.dpkg-old
doorman.toml.pre-proactive-fix
```

### `ls /etc/doorman | grep -c bak`
```
18
```

## 6. Lock glue

### `cat ~/.local/bin/doorman-lock`
```
#!/bin/sh
set -u

log=/tmp/doorman-lock.log
gtk_config="${XDG_CONFIG_HOME:-$HOME/.config}/gtklock/config.ini"
swaylock_config="${XDG_CONFIG_HOME:-$HOME/.config}/swaylock/config"
backend="${DOORMAN_LOCK_BACKEND:-auto}"
watcher="$HOME/.local/bin/doorman-lock-watch"

{
    echo "[$(date -Is)] start pid=$$ user=${USER:-?} display=${WAYLAND_DISPLAY:-?} swaysock=${SWAYSOCK:-?}"
    echo "[$(date -Is)] pre-lock ppid=$PPID session=${XDG_SESSION_ID:-?} $(loginctl show-session "${XDG_SESSION_ID:-}" -p LockedHint 2>&1 | tr '\n' ' ')"
    echo "[$(date -Is)] pre-lock lock-related processes:"
    lock_procs="$(pgrep -a 'gtklock|swaylock|quickshell|dms|qs' 2>&1)"
    if [ -n "$lock_procs" ]; then
        printf '%s\n' "$lock_procs" | sed 's/^/    /'
    else
        echo "    (none)"
    fi

    if [ "$backend" = "auto" ]; then
        if command -v gtklock >/dev/null 2>&1; then
            backend=gtklock
        else
            backend=swaylock
        fi
    fi

    case "$backend" in
        gtklock)
            if ! command -v gtklock >/dev/null 2>&1; then
                echo "[$(date -Is)] gtklock missing"
                exit 1
            fi
            gtklock --daemonize --config "$gtk_config" 2>>"$log"
            rc=$?
            locker=gtklock
            ;;
        swaylock)
            if ! command -v swaylock >/dev/null 2>&1; then
                echo "[$(date -Is)] swaylock missing"
                notify-send "Sway" "No supported screen locker is installed" 2>/dev/null || true
                exit 1
            fi
            if [ -f "$swaylock_config" ]; then
                swaylock --daemonize --config "$swaylock_config" --color 2E3440
            else
                swaylock --daemonize --color 2E3440
            fi
            rc=$?
            locker=swaylock
            ;;
        *)
            echo "[$(date -Is)] unsupported backend=$backend"
            exit 2
            ;;
    esac

    echo "[$(date -Is)] backend=$backend rc=$rc"

    # niri may refuse a second lock client (rc!=0) while a lock is already held;
    # start the watcher anyway if the locker process is alive.
    locker_alive=no
    if [ "$rc" -eq 0 ]; then
        locker_alive=yes
    else
        tries=0
        while [ "$tries" -lt 4 ]; do
            if pgrep -u "${USER:-<user>}" -x "$locker" >/dev/null 2>&1; then
                locker_alive=yes
                break
            fi
            tries=$((tries + 1))
            sleep 0.5
        done
    fi
    echo "[$(date -Is)] watcher decision: rc=$rc ${locker}_alive=$locker_alive"

    if [ "$rc" -eq 0 ] || [ "$locker_alive" = yes ]; then
        "$watcher" "${USER:-<user>}" "$locker" >/dev/null 2>&1 &
        echo "[$(date -Is)] doorman watcher pid=$! locker=$locker"
    fi

    exit "$rc"
} >> "$log" 2>&1
```

### `cat ~/.local/bin/doorman-lock-watch`
```
#!/bin/sh
set -u

log="${DOORMAN_SWAY_WATCH_LOG:-/tmp/sway-doorman-watch.log}"
user="${1:-${USER:-<user>}}"
locker="${2:-swaylock}"
socket="${DOORMAN_SOCKET:-/run/doorman/doorman.sock}"
interval="${DOORMAN_SWAY_POLL_INTERVAL:-2}"

{
    echo "[$(date -Is)] start user=$user locker=$locker socket=$socket"

    attempt=0
    while pgrep -u "$user" -x "$locker" >/dev/null 2>&1; do
        attempt=$((attempt + 1))

        out="$(doorman --socket "$socket" test "$user" 2>&1)"
        rc=$?
        printf '[%s] attempt=%s rc=%s %s\n' "$(date -Is)" "$attempt" "$rc" "$out"

        if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q '^recognized:'; then
            echo "[$(date -Is)] recognized; closing $locker"
            pkill -TERM -u "$user" -x "$locker" >/dev/null 2>&1 || true
            sleep 0.5
            if pgrep -u "$user" -x "$locker" >/dev/null 2>&1; then
                echo "[$(date -Is)] $locker still alive; killing"
                pkill -KILL -u "$user" -x "$locker" >/dev/null 2>&1 || true
            fi
            command -v swaymsg >/dev/null 2>&1 && swaymsg "output * power on" >/dev/null 2>&1 || true
            exit 0
        fi

        sleep "$interval"
    done

    echo "[$(date -Is)] $locker gone; stop"
} >> "$log" 2>&1
```

### `tail -60 /tmp/doorman-lock.log`
```
[2026-11-01T11:03:12+01:00] start pid=3029425 user=<user> display=wayland-1 swaysock=?
[2026-11-01T11:03:13+01:00] backend=gtklock rc=1
[2026-11-01T13:02:16+01:00] start pid=3626844 user=<user> display=wayland-1 swaysock=?
[2026-11-01T13:02:16+01:00] pre-lock ppid=3626843 session=2 LockedHint=no 
[2026-11-01T13:02:16+01:00] pre-lock lock-related processes:
    3561993 dms run
[2026-11-01T13:02:17+01:00] backend=gtklock rc=1
[2026-11-01T13:12:00+01:00] start pid=3668610 user=<user> display=wayland-1 swaysock=?
[2026-11-01T13:12:00+01:00] pre-lock ppid=3668609 session=2 LockedHint=no 
[2026-11-01T13:12:00+01:00] pre-lock lock-related processes:
    3561993 dms run
    3562005 qs -p /usr/share/quickshell/dms/quickshell
[2026-11-01T13:12:01+01:00] backend=gtklock rc=0
[2026-11-01T13:12:01+01:00] watcher decision: rc=0 gtklock_alive=yes
[2026-11-01T13:12:01+01:00] doorman watcher pid=3668738 locker=gtklock
[2026-11-01T19:27:26+01:00] start pid=1373769 user=<user> display=wayland-1 swaysock=?
[2026-11-01T19:27:26+01:00] pre-lock ppid=1373768 session=2 LockedHint=no 
[2026-11-01T19:27:26+01:00] pre-lock lock-related processes:
    3561993 dms run
    3562005 qs -p /usr/share/quickshell/dms/quickshell
[2026-11-01T19:27:27+01:00] backend=gtklock rc=1
[2026-11-01T19:27:27+01:00] watcher decision: rc=1 gtklock_alive=yes
[2026-11-01T19:27:27+01:00] doorman watcher pid=1373853 locker=gtklock
```

## 7. Recent timings

### `journalctl -u doormand --no-pager --since '2 days ago' | grep -iE 'recogni|similarity|liveness|relief|auth' | tail -40`
```
Nov 01 13:02:26 <host> doormand[1859]: 2026-11-01T12:02:26.003496Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.81, frame: 366)
Nov 01 13:02:26 <host> doormand[1859]: 2026-11-01T12:02:26.393281Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.81, frame: 372)
Nov 01 13:02:26 <host> doormand[1859]: 2026-11-01T12:02:26.799751Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.81, frame: 378)
Nov 01 13:02:27 <host> doormand[1859]: 2026-11-01T12:02:27.188470Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.80, frame: 384)
Nov 01 13:02:27 <host> doormand[1859]: 2026-11-01T12:02:27.423649Z  INFO doormand: Pipeline gate closed (no preview clients, disarmed) -> stopping detection/recognition pipeline (idle)
Nov 01 13:12:01 <host> doormand[1859]: 2026-11-01T12:12:01.301678Z  INFO doormand::ipc: Authentication request for user: <user>
Nov 01 13:12:02 <host> doormand[1859]: 2026-11-01T12:12:02.630089Z  INFO doormand: Pipeline gate opened (preview subscribers=0, armed=true) -> starting detection/recognition pipeline
Nov 01 13:12:02 <host> doormand[1859]: 2026-11-01T12:12:02.630179Z  INFO doormand::pipeline::recognition_pipeline: Recognition pipeline started
Nov 01 13:12:02 <host> doormand[1859]: 2026-11-01T12:12:02.955839Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.76, frame: 393)
Nov 01 13:12:03 <host> doormand[1859]: 2026-11-01T12:12:03.164023Z  INFO doormand::ipc: Captured 10 frames for authentication
Nov 01 13:12:03 <host> doormand[1859]: 2026-11-01T12:12:03.357915Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.79, frame: 399)
Nov 01 13:12:03 <host> doormand[1859]: 2026-11-01T12:12:03.757617Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.80, frame: 405)
Nov 01 13:12:03 <host> doormand[1859]: 2026-11-01T12:12:03.911040Z  WARN doormand::ml: Liveness REJECTED (depth gate): relief=0.0491 < threshold=0.0500 — treating as spoof
Nov 01 13:12:03 <host> doormand[1859]: 2026-11-01T12:12:03.911063Z  WARN doormand::ipc: Authentication failed for <user>: liveness/anti-spoof gate rejected
Nov 01 13:12:04 <host> doormand[1859]: 2026-11-01T12:12:04.137616Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.79, frame: 411)
Nov 01 13:12:04 <host> doormand[1859]: 2026-11-01T12:12:04.551771Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.79, frame: 417)
Nov 01 13:12:04 <host> doormand[1859]: 2026-11-01T12:12:04.930636Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.79, frame: 423)
Nov 01 13:12:05 <host> doormand[1859]: 2026-11-01T12:12:05.333164Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.78, frame: 429)
Nov 01 13:12:05 <host> doormand[1859]: 2026-11-01T12:12:05.741206Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.77, frame: 435)
Nov 01 13:12:06 <host> doormand[1859]: 2026-11-01T12:12:05.999744Z  INFO doormand::ipc: Authentication request for user: <user>
Nov 01 13:12:06 <host> doormand[1859]: 2026-11-01T12:12:06.138527Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.76, frame: 441)
Nov 01 13:12:06 <host> doormand[1859]: 2026-11-01T12:12:06.492625Z  INFO doormand::ipc: Captured 10 frames for authentication
Nov 01 13:12:06 <host> doormand[1859]: 2026-11-01T12:12:06.533201Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.78, frame: 447)
Nov 01 13:12:06 <host> doormand[1859]: 2026-11-01T12:12:06.959267Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.79, frame: 453)
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.251434Z  INFO doormand::ipc: Authentication aggregate for <user>: 10 quality frames, cosine=0.7872 (threshold 0.40)
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.294048Z  INFO doormand::ipc: Authentication successful for <user> (similarity: 0.7872)
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.332983Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.80, frame: 459)
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.741419Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.82, frame: 465)
Nov 01 13:12:07 <host> doormand[1859]: 2026-11-01T12:12:07.760169Z  INFO doormand: Pipeline gate closed (no preview clients, disarmed) -> stopping detection/recognition pipeline (idle)
Nov 01 19:27:27 <host> doormand[1859]: 2026-11-01T18:27:27.751688Z  INFO doormand::ipc: Authentication request for user: <user>
Nov 01 19:27:29 <host> doormand[1859]: 2026-11-01T18:27:29.257396Z  INFO doormand: Pipeline gate opened (preview subscribers=0, armed=true) -> starting detection/recognition pipeline
Nov 01 19:27:29 <host> doormand[1859]: 2026-11-01T18:27:29.257589Z  INFO doormand::pipeline::recognition_pipeline: Recognition pipeline started
Nov 01 19:27:29 <host> doormand[1859]: 2026-11-01T18:27:29.570603Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.75, frame: 471)
Nov 01 19:27:29 <host> doormand[1859]: 2026-11-01T18:27:29.795811Z  INFO doormand::ipc: Captured 10 frames for authentication
Nov 01 19:27:29 <host> doormand[1859]: 2026-11-01T18:27:29.961324Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.76, frame: 477)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.383598Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.77, frame: 483)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.641430Z  INFO doormand::ipc: Authentication aggregate for <user>: 10 quality frames, cosine=0.7651 (threshold 0.40)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.705990Z  INFO doormand::ipc: Authentication successful for <user> (similarity: 0.7651)
Nov 01 19:27:30 <host> doormand[1859]: 2026-11-01T18:27:30.784885Z  INFO doormand::pipeline::recognition_pipeline: ✓ User recognized: <user> (similarity: 0.77, frame: 489)
Nov 01 19:27:31 <host> doormand[1859]: 2026-11-01T18:27:31.124971Z  INFO doormand: Pipeline gate closed (no preview clients, disarmed) -> stopping detection/recognition pipeline (idle)
```

## 3b. Extra (model dir, linkage, device perms)

### `ls -l /usr/share/doorman/models /usr/lib/doorman/adapters`
```
/usr/lib/doorman/adapters:
total 0
drwxr-xr-x 1 root root 104 Jun 26 08:56 sway

/usr/share/doorman/models:
total 17188
-rw-r--r-- 1 root root 14826052 Jun 24 17:41 edgeface_s.onnx
-rw-r--r-- 1 root root   232589 Jun 24 17:41 face_detection_yunet_2023mar.onnx
-rw-r--r-- 1 root root  1912594 Jun 24 17:41 minifasnet_v2se.onnx
-rw-r--r-- 1 root root   626197 Jun 24 17:41 minifasnet_v2se_int8.onnx
```

### `ldd /usr/bin/doormand | grep -iE "onnx|ncnn|vulkan|avcodec|gst|v4l"`
```
```

### `ls -l /dev/video0 /dev/video1; id doorman`
```
crw-rw----+ 1 root video 81, 0 Oct 17 18:20 /dev/video0
crw-rw----+ 1 root video 81, 1 Oct 17 18:20 /dev/video1
uid=997(doorman) gid=985(doorman) groups=985(doorman),44(video),991(render)
```

### `readelf -d /usr/bin/doormand | grep -iE "rpath|runpath|NEEDED"`
```
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
```

## Implications for WS1-WS8 (observed facts only; WS numbering not known to collector, so listed by topic)

- Camera: UGREEN Camera 4K (uvcvideo, /dev/video0 capture, /dev/video1 metadata only). MJPG offers 640x480, 1280x720 and 1920x1080 at 30fps (720p/1080p/360p also 60fps). YUYV offers 640x480 and 640x360 at 30fps only; 1280x720 is NOT offered in YUYV, 1280x960 YUYV is 10fps, 1920x1080 YUYV 5fps. Use MJPG for 720p@30.
- Camera controls: auto_exposure menu has only 1 (Manual) and 3 (Aperture Priority; current). exposure_time_absolute range 1-5000 (inactive while auto), gain 0-15 (current 0), backlight_compensation 0-3 (1), power_line_frequency currently 60 Hz (50 Hz also selectable; host timezone is CEST so 50 Hz may be more correct, unverified), focus is continuous auto, brightness/contrast 1-64.
- Backend failure: nokhwa/V4L2 "Failed to open camera" fails in ~0.1-0.4 ms on every start (no detail logged), then falls back to the ffmpeg backend (device 0, 1280x720 @ 30fps) which works. The failure is immediate, so it is not a timeout/format negotiation. The service user doorman is in groups video and render, /dev/video0 is root:video 660, DevicePolicy allows char-video4linux. The error detail needs better logging.
- Service sandbox: no denied/permission/seccomp/apparmor journal hits in 2 days; no SystemCallFilter; RestrictAddressFamilies=AF_UNIX only (no AF_NETLINK, a possible factor for device enumeration, unverified); ProtectHome=true; MemoryMax=2G, TasksMax=128 (current 42 tasks, swap used 353M).
- Unit header says "EP: ncnn" but ExecStart passes --device cpu and doorman.toml has [ml] backend="ort", device="cpu". /proc/PID/maps is unreadable, so the actually loaded libonnxruntime is unconfirmed. doormand is linked only against libstdc++/libgcc/libm/libc (no NEEDED onnxruntime/ffmpeg/gst), so ORT is statically linked or dlopen'd. Candidates: distro libonnxruntime.so.1.21 (1.21.0+dfsg-2, with dnnl provider) or /usr/lib/doorman/libonnxruntime.so (26.8MB, Jun 12) plus a 259MB ROCm provider.
- Models: /usr/share/doorman/models has edgeface_s.onnx, face_detection_yunet_2023mar.onnx, minifasnet_v2se.onnx, minifasnet_v2se_int8.onnx (all Jun 24). Config models_dir=/var/lib/doorman/models, which does not exist in /var/lib/doorman (only dataset/ and embeddings.bin), so the daemon must resolve models elsewhere (config vs reality mismatch).
- CPU: Ryzen 7 8700G, 8C/16T, AVX2, AVX-512 (incl. VNNI, bf16), so int8 ORT models are viable; cpu_threads=0 (ORT chooses).
- Timings: auth request to success is ~3.0-3.4 s end to end (13:12:06.0 -> 13:12:07.29; 19:27:27.75 -> 19:27:30.7), of which camera open/pipeline start is ~1.3-1.5 s (request 19:27:27.75, pipeline started 19:27:29.26). Recognition log lines are ~6 frame ids apart every ~0.4 s. Aggregate cosine 0.765-0.787 vs threshold 0.40; per-frame 0.75-0.82. Detection confidence ~0.91.
- Liveness: one real owner attempt rejected at 13:12:03 by the depth gate: relief=0.0491 < threshold 0.0500 (marginal false reject; the retry 3 s later passed).
- Lock glue: lock script uses gtklock (rc=1 in 3 of 4 recent runs, rc=0 once; watcher still started when gtklock is alive), watcher polls `doorman test` every 2 s and terminates the locker via process signal on a "recognized:" line. Compositor is niri/DMS (dms + quickshell processes), not sway; SWAYSOCK is unset, so the swaymsg step is a no-op. Config has proactive_unlock=false with comment "Sway adapter owns face unlock".
- Config hygiene: 18 *.bak* files in /etc/doorman; the installed unit says "Do not edit by hand".
- Dataset logging is on with keep_raw_frames=true (JPEG quality 92).
