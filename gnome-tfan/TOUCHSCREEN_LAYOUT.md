# Touchscreen Cockpit Layout Design 🖥️📐

Exact positioning specifications for the side touchscreen HUD panel.

## Display Specifications

### Target Resolution
- **Portrait Mode**: 800×1280 or 1080×1920
- **Optimal DPI**: 96-120 for touch targets
- **Minimum touch target**: 48×48 pixels (10mm physical)

### Monitor Placement
```
┌─────────────────┐     ┌──────────────────────────────┐
│  Side Monitor   │     │       Main Monitor           │
│  (Touchscreen)  │     │                              │
│                 │     │   Ara Avatar Display         │
│  T-FAN Cockpit  │     │   + Main Workspace           │
│                 │     │                              │
│  800×1280       │     │       2560×1440              │
└─────────────────┘     └──────────────────────────────┘
```

---

## Layout Zones (800×1280)

### Zone Map
```
┌──────────────────────────────────┐ 0px
│          HUD STRIP               │
│        (Control Buttons)         │
│                                  │
├──────────────────────────────────┤ 200px
│                                  │
│                                  │
│        CONTENT AREA              │
│                                  │
│     (Metrics/Topology/Avatar)    │
│                                  │
│                                  │
│                                  │
│                                  │
├──────────────────────────────────┤ 1180px
│        STATUS BAR                │
└──────────────────────────────────┘ 1280px
```

---

## Zone 1: HUD Control Strip (0-200px)

### Dimensions
- **Position**: x=0, y=0
- **Size**: 800×200px
- **Background**: Linear gradient #0a0e1a → #020306

### Title Section (0-60px)
```
┌──────────────────────────────────┐
│   ⚛️ T-FAN COCKPIT              │  y=20px, font-size=24px
│                                  │  color=#00d4ff
└──────────────────────────────────┘
```

### Button Grid (60-190px)
```
┌────────┬────────┬────────┬────────┐
│OVERVIEW│  GPU   │CPU/RAM │NETWORK │  Row 1: y=70px
│   📊   │   🎮   │   💻   │   🌐   │  Button: 90×75px
├────────┼────────┼────────┼────────┤
│STORAGE │TOPOLOGY│ AVATAR │        │  Row 2: y=150px
│   💾   │   🌌   │   🤖   │        │
└────────┴────────┴────────┴────────┘
```

### Button Specifications
- **Size**: 90×75px each
- **Margin**: 8px between buttons
- **Touch target**: 106×91px (with margins)
- **Grid columns**: 4 columns, 184px each
- **Button positions**:
  - Col 1: x=24px
  - Col 2: x=208px
  - Col 3: x=392px
  - Col 4: x=576px

### Button Styling
```css
.hud-button {
    min-width: 90px;
    min-height: 75px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: 2px solid #00d4ff;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    /* Touch feedback */
    transition: all 0.15s ease;
}

.hud-button:active {
    transform: scale(0.95);
    background: linear-gradient(135deg, #00d4ff, #667eea);
}
```

---

## Zone 2: Content Area (200-1180px)

### Dimensions
- **Position**: x=0, y=200
- **Size**: 800×980px
- **Scrollable**: Yes (touch scroll)

### Padding & Margins
- **Content padding**: 16px all sides
- **Card margin**: 12px
- **Usable width**: 768px

### Content Layouts

#### Overview View
```
┌──────────────────────────────────┐ y=200
│     ┌────────────────────┐       │
│     │  MISSION STATUS    │       │ Status card: 768×150px
│     │  🟢 ALL NOMINAL    │       │ y=216, margin=16px
│     └────────────────────┘       │
│                                  │
│   ┌──────────┬──────────┐        │
│   │   GPU    │   CPU    │        │ Quick cards: 368×120px
│   │   85%    │   42%    │        │ y=382, 2-column grid
│   └──────────┴──────────┘        │
│   ┌──────────┬──────────┐        │
│   │   RAM    │   NET    │        │ y=514
│   │  12 GB   │  15 MB/s │        │
│   └──────────┴──────────┘        │
└──────────────────────────────────┘
```

#### GPU View
```
┌──────────────────────────────────┐
│  ┌────────────────────────────┐  │
│  │ NVIDIA GeForce RTX 3090    │  │ GPU card: 768×280px
│  │                            │  │
│  │        85%                 │  │ Value: font-size=64px
│  │                            │  │
│  │ VRAM: 18432 / 24576 MB     │  │ Details: font-size=18px
│  │ Temp: 72°C                 │  │
│  │ Power: 320W                │  │
│  └────────────────────────────┘  │
│                                  │
│  [Second GPU card if present]    │
└──────────────────────────────────┘
```

#### Topology View (Full Content Area)
```
┌──────────────────────────────────┐
│                                  │
│                                  │
│     WebGL Canvas                 │ Size: 800×980px
│     (Three.js topology)          │ No padding
│                                  │
│     Touch: rotate/zoom           │
│                                  │
│                                  │
└──────────────────────────────────┘
```

#### Avatar View
```
┌──────────────────────────────────┐
│  ┌────────────────────────────┐  │
│  │ ARA AVATAR CONTROL         │  │ Control card: 768×450px
│  │                            │  │
│  │ Profile: [Dropdown      ▼] │  │ Dropdown: 736×48px
│  │                            │  │ Touch target: 48px height
│  │ Style:   [Dropdown      ▼] │  │
│  │                            │  │
│  │ Mood:    [Dropdown      ▼] │  │
│  │                            │  │
│  │ ┌────────────────────────┐ │  │
│  │ │   ✓ APPLY CHANGES      │ │  │ Button: 200×60px
│  │ └────────────────────────┘ │  │
│  │ ┌────────────────────────┐ │  │
│  │ │   💾 SAVE PRESET       │ │  │ Button: 200×60px
│  │ └────────────────────────┘ │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │ CURRENT STATUS             │  │ Status card: 768×200px
│  │ Profile: Professional      │  │
│  │ Style: Realistic           │  │
│  │ Mood: Focused              │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

---

## Zone 3: Status Bar (1180-1280px)

### Dimensions
- **Position**: x=0, y=1180
- **Size**: 800×100px
- **Background**: Semi-transparent #020306cc

### Content
```
┌──────────────────────────────────┐
│  🟢 Connected   |   ⚡ Work Mode │  Status indicators
│  GPU: 72°C      |   12:34 PM     │  font-size: 14px
└──────────────────────────────────┘
```

---

## Touch Interaction Areas

### Tap Zones
```
Zone ID | Position      | Size    | Action
--------|---------------|---------|------------------
HUD-1   | 24,70         | 90×75   | Switch to Overview
HUD-2   | 208,70        | 90×75   | Switch to GPU
HUD-3   | 392,70        | 90×75   | Switch to CPU/RAM
HUD-4   | 576,70        | 90×75   | Switch to Network
HUD-5   | 24,150        | 90×75   | Switch to Storage
HUD-6   | 208,150       | 90×75   | Switch to Topology
HUD-7   | 392,150       | 90×75   | Switch to Avatar
```

### Gesture Zones
```
Zone         | Gesture      | Action
-------------|--------------|------------------
Content Area | Swipe Up/Down| Scroll content
Content Area | Pinch        | Zoom (topology)
Content Area | Two-finger   | Rotate (topology)
Any Card     | Long Press   | Show options
```

### Touch Feedback
```css
/* Ripple effect on tap */
.touch-feedback {
    position: relative;
    overflow: hidden;
}

.touch-feedback::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(0, 212, 255, 0.4);
    transform: scale(0);
    animation: ripple 0.4s ease-out;
}

@keyframes ripple {
    to {
        transform: scale(4);
        opacity: 0;
    }
}
```

---

## Background Video Integration

### Video Layer Setup
```
┌──────────────────────────────────┐
│ Video Layer (z-index: 0)         │
│  • Hologram/robot animation loop │
│  • Opacity: 15-25%               │
│  • Blur: 2px                     │
│  • Desaturated/tinted blue       │
├──────────────────────────────────┤
│ Content Layer (z-index: 1)       │
│  • HUD strip                     │
│  • Metric cards                  │
│  • All UI elements               │
├──────────────────────────────────┤
│ Overlay Layer (z-index: 2)       │
│  • Scanline effect               │
│  • Vignette                      │
│  • Glow effects                  │
└──────────────────────────────────┘
```

### Video Specifications
- **Format**: WebM or MP4 (H.264)
- **Resolution**: 800×1280 (match display)
- **Frame rate**: 24-30fps
- **Duration**: 10-30 second loop
- **File size**: <50MB for smooth playback

### GTK Implementation
```python
# In cockpit_hud.py

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst

class VideoBackground:
    def __init__(self, window):
        Gst.init(None)

        # Create video playback
        self.player = Gst.ElementFactory.make("playbin", "player")
        self.player.set_property("uri", f"file:///path/to/hologram_loop.webm")

        # Create GTK sink for embedding
        self.sink = Gst.ElementFactory.make("gtk4paintablesink", "sink")
        self.player.set_property("video-sink", self.sink)

        # Create picture widget
        self.picture = Gtk.Picture()
        self.picture.set_paintable(self.sink.get_property("paintable"))

        # Style: semi-transparent, blurred
        self.picture.set_opacity(0.2)

        # Enable looping
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._on_eos)

    def _on_eos(self, bus, msg):
        # Loop video
        self.player.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            0
        )

    def play(self):
        self.player.set_state(Gst.State.PLAYING)

    def stop(self):
        self.player.set_state(Gst.State.NULL)
```

### CSS Overlay Effects
```css
/* Scanline overlay */
.cockpit-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 100;
}

.cockpit-overlay::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 212, 255, 0.02) 2px,
        rgba(0, 212, 255, 0.02) 4px
    );
}

/* Vignette effect */
.cockpit-overlay::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(
        ellipse at center,
        transparent 60%,
        rgba(0, 0, 0, 0.4) 100%
    );
}
```

### Recommended Video Content

1. **Hologram Grid** - Rotating 3D wireframe grid
2. **Data Streams** - Flowing particle trails
3. **Robot Silhouettes** - Subtle robot/android outlines
4. **Circuit Patterns** - Animated circuit board traces
5. **Neural Networks** - Pulsing node connections

**Sources:**
- Create in Blender/After Effects
- Purchase from stock (Envato, Artgrid)
- Generate with AI (RunwayML, Pika)

---

## Responsive Scaling

### 1080×1920 Layout (Full HD Portrait)
Scale all dimensions by 1.35x:

```
Zone           | 800×1280    | 1080×1920
---------------|-------------|-------------
HUD Strip      | 200px       | 270px
Button Grid    | 90×75px     | 122×101px
Content Area   | 980px       | 1323px
Status Bar     | 100px       | 135px
```

### Touch Target Scaling
```css
/* Scale touch targets for higher DPI */
@media (min-height: 1920px) {
    .hud-button {
        min-width: 122px;
        min-height: 101px;
        font-size: 16px;
    }

    .metric-value-huge {
        font-size: 86px;
    }
}
```

---

## Performance Optimization

### Video Background
- Use hardware-accelerated decoding (VA-API/NVDEC)
- Reduce resolution if GPU-bound
- Pause video when topology visualization active

### Metrics Updates
- Batch DOM updates
- Use GPU compositing where possible
- Throttle updates to 2Hz for non-critical metrics

### Touch Responsiveness
- Target <100ms touch response
- Use CSS transforms (GPU-accelerated)
- Avoid layout thrashing on scroll

---

## Implementation Checklist

### Core Layout
- [x] HUD strip with button grid
- [x] Content area with scroll
- [x] Status bar
- [ ] Video background layer
- [ ] Scanline/vignette overlays

### Touch Interactions
- [x] Tap to switch views
- [ ] Swipe to scroll
- [ ] Pinch to zoom (topology)
- [ ] Long press for options
- [ ] Ripple feedback effect

### Views
- [x] Overview (mission status)
- [x] GPU metrics
- [x] CPU/RAM metrics
- [x] Network metrics
- [x] Storage metrics
- [x] Topology visualization
- [x] Avatar controls

### Polish
- [ ] Smooth animations
- [ ] Loading states
- [ ] Error states
- [ ] Empty states
- [ ] Transition effects

---

## Files to Create/Modify

1. **`cockpit_hud.py`** - Main app (done)
2. **`cockpit_theme.css`** - Extract CSS to file
3. **`video_background.py`** - GStreamer video player
4. **`touch_gestures.py`** - Gesture recognizers
5. **`assets/hologram_loop.webm`** - Background video

---

**Your personal mission control - pixel perfect.** 🚀📐✨
