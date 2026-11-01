# Advanced Super Sway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished Nord-themed Sway desktop layer with launcher, top status panel, notifications, lock/idle, screenshots, power menu, and per-window layout rules.

**Architecture:** Keep Sway as the compositor and split user config by responsibility under `~/.config/sway/conf.d`. Use Waybar for the top panel, Fuzzel for launching apps, Mako for notifications, Swaylock/Swayidle for lock behavior, Wlogout for session actions, and small shell scripts for repeated desktop actions.

**Tech Stack:** Sway, Waybar, Fuzzel, Mako, Swaylock, Swayidle, Grim, Slurp, wl-clipboard, Wlogout, Kanshi, Foot/WezTerm, PipeWire/PulseAudio CLI tools.

---

### Task 1: Install Runtime Packages

**Files:**
- No repo files modified.

- [ ] **Step 1: Install desktop components**

Run:

```bash
sudo apt-get update
sudo apt-get install -y waybar fuzzel mako-notifier swaylock swayidle grim slurp wl-clipboard wlogout kanshi brightnessctl playerctl pamixer jq network-manager-gnome blueman pavucontrol xdg-desktop-portal-wlr
```

Expected: packages install successfully. If a package is missing, use `apt-cache search` to identify the Ubuntu package name.

### Task 2: Create Config File Layout

**Files:**
- Create: `~/.config/sway/config`
- Create: `~/.config/sway/conf.d/autostart.conf`
- Create: `~/.config/sway/conf.d/bindings.conf`
- Create: `~/.config/sway/conf.d/input.conf`
- Create: `~/.config/sway/conf.d/layout.conf`
- Create: `~/.config/sway/conf.d/theme.conf`
- Create: `~/.config/waybar/config`
- Create: `~/.config/waybar/style.css`
- Create: `~/.config/fuzzel/fuzzel.ini`
- Create: `~/.config/mako/config`
- Create: `~/.config/swaylock/config`
- Create: `~/.config/wlogout/layout`
- Create: `~/.config/wlogout/style.css`

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p ~/.config/sway/conf.d ~/.config/waybar ~/.config/fuzzel ~/.config/mako ~/.config/swaylock ~/.config/wlogout ~/.config/kanshi ~/.local/bin
```

Expected: all directories exist.

- [ ] **Step 2: Write configs**

Write files with focused responsibilities. `~/.config/sway/config` should only set core variables and include `conf.d/*.conf`.

### Task 3: Create Helper Scripts

**Files:**
- Create: `~/.local/bin/sway-shot`
- Create: `~/.local/bin/sway-menu`
- Create: `~/.local/bin/sway-window-menu`
- Create: `~/.local/bin/sway-lock`
- Create: `~/.local/bin/sway-audio`

- [ ] **Step 1: Write scripts**

Scripts must be POSIX shell compatible and executable.

- [ ] **Step 2: Verify scripts**

Run:

```bash
sh -n ~/.local/bin/sway-shot ~/.local/bin/sway-menu ~/.local/bin/sway-window-menu ~/.local/bin/sway-lock ~/.local/bin/sway-audio
```

Expected: no syntax errors.

### Task 4: Validate and Smoke Test

**Files:**
- Read all created configs.

- [ ] **Step 1: Validate Sway config**

Run:

```bash
sway --validate --config ~/.config/sway/config
```

Expected: exit code 0.

- [ ] **Step 2: Validate JSON configs**

Run:

```bash
jq empty ~/.config/waybar/config ~/.config/wlogout/layout
```

Expected: exit code 0.

- [ ] **Step 3: Report launch commands**

Provide the user with:

```bash
sway --config ~/.config/sway/config
systemctl --user status xdg-desktop-portal-wlr.service
```
