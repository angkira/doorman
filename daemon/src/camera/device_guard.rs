//! Refuse virtual capture devices and resolve the camera node.
//!
//! v4l2loopback-style nodes are writable by the `video` group, so any member
//! can feed arbitrary frames to the daemon through them (docs/THREAT-MODEL.md,
//! "Camera device"). This module classifies a device from its driver, card,
//! bus info and sysfs facts, and resolves `device_index` / `device_path` to a
//! `/dev/videoN` node. The classification is a pure function; sysfs and device
//! node access are injectable for tests.
//!
//! Limits: a root attacker, or a malicious USB device that reports a
//! physical-looking identity, is not detected. The sysfs checks used for the
//! ffmpeg and GStreamer backends are best effort; the native backend also
//! checks the VIDIOC_QUERYCAP strings.

use anyhow::{anyhow, bail, Result};
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

/// Directory with one entry per video4linux node.
pub const SYSFS_VIDEO4LINUX: &str = "/sys/class/video4linux";

/// Where device nodes live; a pinned `device_path` must resolve inside it.
pub const DEV_ROOT: &str = "/dev";

/// File-name prefix of a V4L2 capture node.
const NODE_PREFIX: &str = "video";

/// Lower-case, alphanumeric-only substrings that identify a virtual driver,
/// card name, bus info or sysfs name ("v4l2 loopback", "v4l2loopback-000" and
/// "V4L2_Loopback" all normalise to contain "v4l2loopback").
const VIRTUAL_MARKERS: &[&str] = &[
    "v4l2loopback",
    "akvcam",
    "vivid",
    "vimc",
    "virtual",
    "droidcam",
    "dummy",
];

/// What the kernel reports about one node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceFacts {
    /// VIDIOC_QUERYCAP `driver` (native backend only).
    pub driver: Option<String>,
    /// VIDIOC_QUERYCAP `card`.
    pub card: Option<String>,
    /// VIDIOC_QUERYCAP `bus_info`.
    pub bus_info: Option<String>,
    /// `/sys/class/video4linux/videoN/*`, when readable.
    pub sysfs: Option<SysfsFacts>,
}

/// Facts read from `/sys/class/video4linux/videoN`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SysfsFacts {
    /// Contents of `name`.
    pub name: Option<String>,
    /// Whether `device` (the parent device) exists.
    pub has_parent: bool,
    /// Base name of the `device/driver` symlink target.
    pub driver: Option<String>,
    /// Base name of the `device/subsystem` symlink target (usb, pci, platform).
    pub subsystem: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceClass {
    Physical,
    /// Refused; the string says why.
    Virtual(String),
}

fn normalise(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn marker_in(field: &str, value: Option<&str>) -> Option<String> {
    let v = value?;
    let n = normalise(v);
    VIRTUAL_MARKERS
        .iter()
        .find(|m| n.contains(*m))
        .map(|m| format!("{field} {v:?} matches virtual-device marker {m:?}"))
}

/// Classify a device. Any virtual marker in any identity string, or a sysfs
/// entry without a parent device, makes it virtual.
pub fn classify(facts: &DeviceFacts) -> DeviceClass {
    let sys = facts.sysfs.as_ref();
    let checks = [
        marker_in("driver", facts.driver.as_deref()),
        marker_in("card", facts.card.as_deref()),
        marker_in("bus_info", facts.bus_info.as_deref()),
        marker_in("sysfs name", sys.and_then(|s| s.name.as_deref())),
        marker_in("sysfs driver", sys.and_then(|s| s.driver.as_deref())),
    ];
    if let Some(reason) = checks.into_iter().flatten().next() {
        return DeviceClass::Virtual(reason);
    }
    if let Some(s) = sys {
        if !s.has_parent {
            return DeviceClass::Virtual("sysfs node has no parent bus device".to_string());
        }
    }
    DeviceClass::Physical
}

/// Read sysfs facts for `video<index>` under `root`; `None` when the node is
/// not present there.
pub fn read_sysfs_facts(root: &Path, index: u32) -> Option<SysfsFacts> {
    let node = root.join(format!("{NODE_PREFIX}{index}"));
    if !node.exists() {
        return None;
    }
    let link_base = |p: PathBuf| {
        std::fs::read_link(p)
            .ok()
            .and_then(|t| t.file_name().map(|n| n.to_string_lossy().into_owned()))
    };
    Some(SysfsFacts {
        name: std::fs::read_to_string(node.join("name"))
            .ok()
            .map(|s| s.trim().to_string()),
        has_parent: node.join("device").exists(),
        driver: link_base(node.join("device/driver")),
        subsystem: link_base(node.join("device/subsystem")),
    })
}

/// Device node inspection, injectable for tests.
pub trait NodeFs {
    fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf>;
    fn is_char_device(&self, path: &Path) -> bool;
}

pub struct RealNodeFs;

impl NodeFs for RealNodeFs {
    fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf> {
        std::fs::canonicalize(path)
    }
    fn is_char_device(&self, path: &Path) -> bool {
        std::fs::metadata(path).is_ok_and(|m| m.file_type().is_char_device())
    }
}

/// The node a camera config selects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraNode {
    pub index: u32,
    pub path: PathBuf,
}

/// Parse `/dev/video<N>` (directly under `dev_root`) into its index.
fn node_index(path: &Path, dev_root: &Path) -> Option<u32> {
    if path.parent()? != dev_root {
        return None;
    }
    path.file_name()?
        .to_str()?
        .strip_prefix(NODE_PREFIX)?
        .parse()
        .ok()
}

/// Resolve the configured camera: `device_path` (a stable symlink such as
/// `/dev/v4l/by-id/...`) if set, else `/dev/video<device_index>`.
pub fn resolve_node(
    device_path: Option<&str>,
    device_index: u32,
    dev_root: &Path,
    fs: &dyn NodeFs,
) -> Result<CameraNode> {
    let Some(configured) = device_path else {
        return Ok(CameraNode {
            index: device_index,
            path: dev_root.join(format!("{NODE_PREFIX}{device_index}")),
        });
    };
    let real = fs
        .canonicalize(Path::new(configured))
        .map_err(|e| anyhow!("camera.device_path {configured:?} cannot be resolved: {e}"))?;
    let index = node_index(&real, dev_root).ok_or_else(|| {
        anyhow!(
            "camera.device_path {configured:?} resolves to {}, which is not {}/{NODE_PREFIX}<N>",
            real.display(),
            dev_root.display()
        )
    })?;
    if !fs.is_char_device(&real) {
        bail!(
            "camera.device_path {configured:?} resolves to {}, which is not a character device",
            real.display()
        );
    }
    Ok(CameraNode { index, path: real })
}

/// Refuse a virtual device unless `allow_virtual`.
pub fn enforce(facts: &DeviceFacts, node: &CameraNode, allow_virtual: bool) -> Result<()> {
    match classify(facts) {
        DeviceClass::Physical => Ok(()),
        DeviceClass::Virtual(why) if allow_virtual => {
            tracing::warn!(
                "camera {} looks virtual ({why}); accepted because camera.allow_virtual_devices = true",
                node.path.display()
            );
            Ok(())
        }
        DeviceClass::Virtual(why) => bail!(
            "refusing virtual camera {}: {why}; set camera.allow_virtual_devices = true to override (testing only)",
            node.path.display()
        ),
    }
}

/// Best-effort check from sysfs alone (ffmpeg and GStreamer backends). A node
/// that is absent from sysfs cannot be verified and is refused.
pub fn enforce_sysfs(node: &CameraNode, sysfs_root: &Path, allow_virtual: bool) -> Result<()> {
    let Some(sysfs) = read_sysfs_facts(sysfs_root, node.index) else {
        if allow_virtual {
            return Ok(());
        }
        bail!(
            "cannot verify camera {}: {}/{NODE_PREFIX}{} is not readable; refusing (camera.allow_virtual_devices = true skips the check)",
            node.path.display(),
            sysfs_root.display(),
            node.index
        );
    };
    enforce(
        &DeviceFacts {
            sysfs: Some(sysfs),
            ..Default::default()
        },
        node,
        allow_virtual,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn facts(driver: &str, card: &str, bus: &str) -> DeviceFacts {
        DeviceFacts {
            driver: Some(driver.into()),
            card: Some(card.into()),
            bus_info: Some(bus.into()),
            sysfs: None,
        }
    }

    fn usb_sysfs(name: &str) -> SysfsFacts {
        SysfsFacts {
            name: Some(name.into()),
            has_parent: true,
            driver: Some("uvcvideo".into()),
            subsystem: Some("usb".into()),
        }
    }

    #[test]
    fn querycap_classification_table() {
        let table = [
            ("uvcvideo", "Integrated Camera", "usb-0000:00:14.0-8", false),
            (
                "uvcvideo",
                "HD Pro Webcam C920",
                "usb-0000:00:14.0-1",
                false,
            ),
            ("unicam", "unicam", "platform:3f801000.csi", false),
            (
                "v4l2 loopback",
                "Dummy video device (0x0000)",
                "platform:v4l2loopback-000",
                true,
            ),
            (
                "v4l2 loopback",
                "OBS Virtual Camera",
                "platform:v4l2loopback-001",
                true,
            ),
            ("v4l2loopback", "x", "x", true),
            ("uvcvideo", "x", "platform:V4L2Loopback-002", true),
            ("vivid", "vivid", "platform:vivid-000", true),
            ("akvcam", "AkVCam", "platform:akvcam", true),
            ("uvcvideo", "DroidCam Video", "usb-1", true),
        ];
        for (driver, card, bus, virt) in table {
            let got = matches!(classify(&facts(driver, card, bus)), DeviceClass::Virtual(_));
            assert_eq!(got, virt, "{driver} / {card} / {bus}");
        }
    }

    #[test]
    fn sysfs_classification_table() {
        let physical = DeviceFacts {
            sysfs: Some(usb_sysfs("Integrated Camera: Integrated C")),
            ..Default::default()
        };
        assert_eq!(classify(&physical), DeviceClass::Physical);

        let loop_name = DeviceFacts {
            sysfs: Some(SysfsFacts {
                name: Some("OBS Virtual Camera".into()),
                ..usb_sysfs("")
            }),
            ..Default::default()
        };
        assert!(matches!(classify(&loop_name), DeviceClass::Virtual(_)));

        let loop_driver = DeviceFacts {
            sysfs: Some(SysfsFacts {
                driver: Some("v4l2loopback".into()),
                subsystem: Some("platform".into()),
                ..usb_sysfs("Cam")
            }),
            ..Default::default()
        };
        assert!(matches!(classify(&loop_driver), DeviceClass::Virtual(_)));

        let no_parent = DeviceFacts {
            sysfs: Some(SysfsFacts {
                name: Some("Cam".into()),
                has_parent: false,
                driver: None,
                subsystem: None,
            }),
            ..Default::default()
        };
        assert!(matches!(classify(&no_parent), DeviceClass::Virtual(_)));

        // A platform SoC camera with a parent and no marker stays allowed.
        let soc = DeviceFacts {
            sysfs: Some(SysfsFacts {
                subsystem: Some("platform".into()),
                driver: Some("unicam".into()),
                ..usb_sysfs("unicam-image")
            }),
            ..Default::default()
        };
        assert_eq!(classify(&soc), DeviceClass::Physical);
    }

    fn node() -> CameraNode {
        CameraNode {
            index: 0,
            path: PathBuf::from("/dev/video0"),
        }
    }

    #[test]
    fn enforce_refuses_virtual_unless_allowed() {
        let f = facts("v4l2 loopback", "Cam", "platform:v4l2loopback-000");
        let err = enforce(&f, &node(), false).unwrap_err().to_string();
        assert!(err.contains("refusing virtual camera"), "{err}");
        enforce(&f, &node(), true).unwrap();
        enforce(&facts("uvcvideo", "Cam", "usb-1"), &node(), false).unwrap();
    }

    fn fake_sysfs(dir: &Path, index: u32, name: &str, driver: Option<&str>, parent: bool) {
        let n = dir.join(format!("video{index}"));
        std::fs::create_dir_all(&n).unwrap();
        std::fs::write(n.join("name"), format!("{name}\n")).unwrap();
        if parent {
            let dev = dir.join(format!("parent{index}"));
            std::fs::create_dir_all(&dev).unwrap();
            std::os::unix::fs::symlink(&dev, n.join("device")).unwrap();
            if let Some(d) = driver {
                std::os::unix::fs::symlink(
                    format!("../../../bus/x/drivers/{d}"),
                    dev.join("driver"),
                )
                .unwrap();
            }
            std::os::unix::fs::symlink("../../../bus/usb", dev.join("subsystem")).unwrap();
        }
    }

    #[test]
    fn sysfs_root_is_injectable_and_enforced() {
        let dir = std::env::temp_dir().join(format!("doorman-guard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        fake_sysfs(&dir, 0, "Integrated Camera", Some("uvcvideo"), true);
        fake_sysfs(&dir, 1, "Dummy video device", Some("v4l2loopback"), true);
        fake_sysfs(&dir, 2, "Cam", None, false);
        let n = |i| CameraNode {
            index: i,
            path: PathBuf::from(format!("/dev/video{i}")),
        };
        enforce_sysfs(&n(0), &dir, false).unwrap();
        assert!(enforce_sysfs(&n(1), &dir, false).is_err());
        enforce_sysfs(&n(1), &dir, true).unwrap();
        assert!(enforce_sysfs(&n(2), &dir, false).is_err());
        // Not present in sysfs: unverifiable, refused unless overridden.
        assert!(enforce_sysfs(&n(7), &dir, false).is_err());
        enforce_sysfs(&n(7), &dir, true).unwrap();
        let facts = read_sysfs_facts(&dir, 0).unwrap();
        assert_eq!(facts.driver.as_deref(), Some("uvcvideo"));
        assert_eq!(facts.subsystem.as_deref(), Some("usb"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[derive(Default)]
    struct FakeFs {
        links: HashMap<PathBuf, PathBuf>,
        chars: Vec<PathBuf>,
    }

    impl NodeFs for FakeFs {
        fn canonicalize(&self, p: &Path) -> std::io::Result<PathBuf> {
            self.links
                .get(p)
                .cloned()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        }
        fn is_char_device(&self, p: &Path) -> bool {
            self.chars.iter().any(|c| c == p)
        }
    }

    #[test]
    fn resolve_node_by_index_and_by_id() {
        let dev = Path::new(DEV_ROOT);
        let fs = FakeFs::default();
        let by_index = resolve_node(None, 3, dev, &fs).unwrap();
        assert_eq!(by_index.path, PathBuf::from("/dev/video3"));

        let by_id = "/dev/v4l/by-id/usb-Cam-video-index0";
        let mut fs = FakeFs::default();
        fs.links
            .insert(PathBuf::from(by_id), PathBuf::from("/dev/video4"));
        fs.chars.push(PathBuf::from("/dev/video4"));
        let got = resolve_node(Some(by_id), 0, dev, &fs).unwrap();
        assert_eq!(got.index, 4);
        assert_eq!(got.path, PathBuf::from("/dev/video4"));
    }

    #[test]
    fn resolve_node_rejects_bad_device_paths() {
        let dev = Path::new(DEV_ROOT);
        let mut fs = FakeFs::default();
        for (link, target) in [
            ("/dev/v4l/by-id/a", "/dev/null"),
            ("/dev/v4l/by-id/b", "/tmp/video0"),
            ("/dev/v4l/by-id/c", "/dev/videoX"),
            ("/dev/v4l/by-id/d", "/dev/sub/video0"),
            ("/dev/v4l/by-id/regular", "/dev/video9"),
        ] {
            fs.links.insert(PathBuf::from(link), PathBuf::from(target));
        }
        for link in ["a", "b", "c", "d", "regular", "missing"] {
            let p = format!("/dev/v4l/by-id/{link}");
            assert!(resolve_node(Some(&p), 0, dev, &fs).is_err(), "{link}");
        }
    }
}
