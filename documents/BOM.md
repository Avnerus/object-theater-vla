# Bill of Materials — CTAM Tendon Robot + TPU Gripper

Based on:
- Hansen et al. (2026) "Tendon-Actuated Robots with a Tapered, Flexible Polymer Backbone: Design, Fabrication, and Modeling" (arXiv:2603.19124)
- Hansen et al. (2026) "The quadruped soft tail: Compliant grasping and swabbing for contamination surveys in harsh environments" (arXiv:2606.30900)

---

## 1. 3D-Printed Parts (TPU 95A)

All printed via FDM. TPU 95A for backbone + gripper; PLA/PETG optional for rigid discs.

| # | Item | Qty | Notes |
|---|------|-----|-------|
| 1a | Tapered hollow backbone (TPU 95A) | 1 | ~405 mm length, tapered from base to tip, fixed inner diameter (~4.5 mm for endoscope/tool routing). Parametric CAD via Inventor iLogic scripts (paper Appendix). |
| 1b | Rigid discs, friction-fit (TPU or PLA) | ~20–30 | Disc count depends on design. Tapered radii, logarithmic spacing. Each has 3 equidistant tendon-routing holes + 1 central tool hole. Lofted between holes for mass reduction. |
| 1c | Soft gripper — 6-finger, Φ=20° (TPU 95A) | 1 | 20% infill recommended. Helical tendon routing, single traverse per finger. Flat interior edges angled 20°. |
| 1d | Electronics base housing (PLA/PETG) | 1 | Hollow cylindrical enclosure, houses 3 motors + load cells + spools. Mounting flange for quadruped/arm. |
| 1e | Motor mount hinges & spools | 3 | Printed as part of actuation assembly. |
| 1f | Gripper base / tool coupling | 1 | Couples gripper tendon to backbone central channel. |

**TPU Filament:**
- TPU 95A (Shore A), 1.75 mm diameter — ~1–2 kg total (backbone + discs + gripper + spares)
- Recommended: eSun eFlex TPU 95A, Prusa TPU 95A, or similar

---

## 2. Actuation & Sensing

| # | Item | Qty | Specs |
|---|------|-----|-------|
| 2a | Dynamixel XH430-210T servo motors | 3 | 2.0 A max current, 3.5 kg·cm stall torque, 210°/s speed, 1024-bit resolution. ~$120 each. |
| 2b | FX29 compression load cells | 3 | 5 kg capacity, 0–5 kg range, analog output. ~$25–35 each. |
| 2c | Motor spools (3D-printed or machined) | 3 | Custom, sized for tendon diameter + ~20 wraps. |
| 2d | Hinge pins / bearings | 3 | For motor-to-load-cell force transmission. |

---

## 3. Tendons & Cabling

| # | Item | Qty | Specs |
|---|------|-----|-------|
| 3a | Main actuation tendons (Fishing line or Kevlar) | 3 | 0.5–1.0 mm diameter, high tensile strength, low stretch. ~405 mm each + spool margin. |
| 3b | Gripper actuation tendon | 1 | Same material, routed through backbone center to gripper. Helical routing in gripper. |
| 3c | Dynamixel power/data cable (5-pin) | 3 | Standard DXL 5-pin JST-SH cables, appropriate length. |
| 3d | Load cell wiring | 3 | Shielded 4-wire (excitation + signal). |
| 3e | Endoscopic camera cable (optional) | 1 | For 4.5 mm endoscope routing through backbone. |

---

## 4. Electronics & Control

| # | Item | Qty | Specs |
|---|------|-----|-------|
| 4a | Dynamixel controller / USB2DXL | 1 | Dynamixel2Arduino or OpenCR board for motor control. ~$50–80. |
| 4b | Microcontroller (Arduino/Raspberry Pi) | 1 | Main controller. RPi Zero 2W or Raspberry Pi 4 for WiFi teleoperation. |
| 4c | ADC for load cells (HX711 or similar) | 1 | 24-bit, 3-channel (one per load cell). ~$5. |
| 4d | Power distribution board | 1 | 12V in, 5V/12V out for motors + logic. |
| 4e | LiPo battery / power supply | 1 | 11.1V 2200mAh or similar, sized for motor current draw. |
| 4f | WiFi router (optional) | 1 | For remote teleoperation over UDP (per paper). |

---

## 5. Mechanical Hardware

| # | Item | Qty | Specs |
|---|------|-----|-------|
| 5a | M3 screws, nuts, washers | assorted | For disc assembly, motor mounts, housing. |
| 5b | M2.5 screws | assorted | For electronics mounting. |
| 5c | Mounting plate / bracket | 1 | Interfaces CTAM base to quadruped or rigid arm. |
| 5d | Heat shrink tubing | 1 roll | For wire protection. |
| 5e | Cable ties / zip ties | assorted | For cable management. |

---

## 6. Optional / Endoscopic Integration

| # | Item | Qty | Specs |
|---|------|-----|-------|
| 6a | 4.5 mm endoscopic camera | 1 | Borescope, rigid or semi-flexible. Routed through backbone center channel. |
| 6b | Gripper tool / swab attachment | 1 | For contamination survey application. |

---

## 7. Tools & Consumables

| # | Item | Qty | Notes |
|---|------|-----|-------|
| 7a | 3D printer (FDM, TPU-capable) | 1 | Direct drive extruder recommended for TPU. |
| 7b | Glue / adhesive | 1 | Super glue for tendon anchoring to distal disc. |
| 7c | Sandpaper / files | 1 set | For post-processing printed parts. |
| 7d | Multimeter | 1 | For load cell / wiring verification. |

---

## Estimated Cost Breakdown (USD)

| Category | Low | High |
|----------|-----|------|
| TPU filament | $30 | $60 |
| Dynamixel XH430-210T × 3 | $360 | $400 |
| FX29 load cells × 3 | $75 | $105 |
| Microcontroller + DXL controller | $70 | $130 |
| HX711 ADC + electronics | $15 | $30 |
| Tendons (fishing line/Kevlar) | $10 | $25 |
| Mechanical hardware | $20 | $40 |
| Power supply / battery | $25 | $50 |
| Optional endoscope | $30 | $100 |
| **TOTAL** | **~$635** | **~$940** |

*Excludes 3D printer cost (assumed available).*

---

## Key Design Parameters (from papers)

- **Backbone length:** 405 mm
- **Backbone material:** TPU 95A (FDM printed)
- **Taper:** Logarithmic spiral when fully curled
- **Disc spacing:** Logarithmic ratio (parametric via iLogic)
- **Tendon count:** 3 main + 1 gripper
- **Gripper:** 6 fingers, Φ=20°, 20% infill, 1 traverse
- **Gripper actuation force:** ~10 N (at 20% infill)
- **Total system mass:** ~900 g (with all components)
- **Max height from mount:** 183 mm (folded)
- **Motors:** Dynamixel XH430-210T (×3)
- **Load cells:** FX29, 5 kg (×3)

---

## Notes

1. The paper authors provide parametric Inventor iLogic scripts for CAD generation (Appendix of arXiv:2603.19124). These automate disc sizing, spacing, and hole placement from a spreadsheet of parameters.
2. TPU printing requires a direct-drive extruder and heated bed (~50-60°C). Print speed ~20-30 mm/s, nozzle ~220-230°C.
3. The gripper is a single 3D-printed part — very simple fabrication.
4. For the gripper tendon, the paper recommends a single helical traverse through each finger for best shape recovery.
5. The backbone is hollow with a fixed inner diameter to route the gripper tendon (and optionally an endoscope) through the center.
6. Model calibration requires a line search on Young's modulus to match physical behavior (FDM-printed TPU has variable E).
