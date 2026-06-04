# Vignesh Guru Prasath HL7 Viewer & Editor

A free, open-source HL7 message viewer and editor for internal/personal use.
No license restrictions. Free for everyone.

---

## Features

- **Paste & Parse** — Paste any HL7 v2.x message and instantly see a structured breakdown
- **Segment Browser** — Tree view of all segments (MSH, PID, OBR, OBX, PV1, NTE, etc.)
- **Field-level editing** — Click any field, edit its value, press Enter to apply
- **Component breakdown** — See ^ delimited components and & sub-components separately
- **Repetition support** — ~ repetitions are shown and counted
- **Input / Output tabs** — Original input and rebuilt/edited output side by side
- **Open & Save files** — Load `.hl7`, `.txt`, `.msg` files; save edited output
- **Copy output** — One-click copy of the rebuilt HL7 message
- **Sample message** — Built-in ORU^R01 sample to explore the tool
- **Field descriptions** — Hover or select MSH, PID, OBX, OBR, PV1 fields to see what they mean

---

## Requirements

- Python 3.8+ (Tkinter is included by default on Windows/macOS)
- On Linux: `sudo apt-get install python3-tk`

---

## Run

```bash
python hl7_viewer v3.py
```

Or double-click the script on Windows/macOS.

---

## Usage

1. **Paste** your HL7 message in the Input tab
2. Click **▶ Parse Message** (or press `Ctrl+Enter`)
3. Browse segments in the **Segment Browser** on the right
4. Click any field row to see its value and component breakdown below
5. Edit the field value in the **Field Editor** and press Enter / **Apply**
6. The **Output tab** shows the rebuilt message in real time

---

## Supported Segments

MSH, EVN, PID, PV1, PV2, OBR, OBX, ORC, NK1, DG1, GT1, IN1, IN2, AL1, PRB, ROL, NTE, MRG, FT1, TXA, RXA, RXO, RXR, TQ1, SFT, SPM, SAC, MSA, ERR, STF, MFI, MFE, and Z-segments.

---

## License

**No license. Free for everyone.**
This tool is open-source and intended for personal/internal use.
You may use, copy, modify, and share it freely.

---

## Contributing

Pull requests welcome at: https://github.com/vigneshguruprasath90/AIProject

---

*Built with Python + Tkinter. No external dependencies required.*
