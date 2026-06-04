#!/usr/bin/env python3
"""
HL7 Viewer & Editor — Enhanced
-------------------------------
Features:
  • Segment browser with field-level editing (original)
  • Input / Output tabs with rebuild (original)
  • MSH-1/MSH-2 correct field numbering (fixed)
  • HL7 Inspector-style features: segment filter, message statistics,
    field search, OBX base64 PDF viewer, HL7 message compare tool
No license restrictions. Free for everyone.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import re
import json
import base64
import tempfile
import os
import webbrowser
import difflib
from datetime import datetime


HL7_SEGMENT_DESCRIPTIONS = {
    "MSH": "Message Header",
    "EVN": "Event Type",
    "PID": "Patient Identification",
    "PV1": "Patient Visit",
    "PV2": "Patient Visit - Additional Info",
    "OBR": "Observation Request",
    "OBX": "Observation/Result",
    "ORC": "Common Order",
    "NK1": "Next of Kin",
    "DG1": "Diagnosis",
    "GT1": "Guarantor",
    "IN1": "Insurance",
    "IN2": "Insurance - Additional Info",
    "AL1": "Patient Allergy",
    "PRB": "Problem",
    "ROL": "Role",
    "TXA": "Transcription Document Header",
    "FT1": "Financial Transaction",
    "ZPD": "Custom Patient Demographic (Z-Segment)",
    "ZIN": "Custom Insurance (Z-Segment)",
    "MSA": "Message Acknowledgment",
    "ERR": "Error",
    "QRD": "Query Definition",
    "SPM": "Specimen",
    "SAC": "Specimen Container Detail",
    "NTE": "Notes and Comments",
    "MRG": "Merge Patient Info",
    "MFE": "Master File Entry",
    "MFI": "Master File Identification",
    "STF": "Staff Identification",
    "RXA": "Pharmacy/Treatment Administration",
    "RXO": "Pharmacy/Treatment Order",
    "RXR": "Pharmacy/Treatment Route",
    "TQ1": "Timing/Quantity",
    "SFT": "Software Segment",
}

MSH_FIELDS = {
    1: "Field Separator",
    2: "Encoding Characters",
    3: "Sending Application",
    4: "Sending Facility",
    5: "Receiving Application",
    6: "Receiving Facility",
    7: "Date/Time of Message",
    8: "Security",
    9: "Message Type",
    10: "Message Control ID",
    11: "Processing ID",
    12: "Version ID",
    13: "Sequence Number",
    14: "Continuation Pointer",
    15: "Accept Acknowledgment Type",
    16: "Application Acknowledgment Type",
    17: "Country Code",
    18: "Character Set",
    19: "Principal Language of Message",
    20: "Alternate Character Set Handling",
    21: "Message Profile Identifier",
}

PID_FIELDS = {
    1: "Set ID",
    2: "Patient ID (External)",
    3: "Patient Identifier List",
    4: "Alternate Patient ID",
    5: "Patient Name",
    6: "Mother's Maiden Name",
    7: "Date/Time of Birth",
    8: "Administrative Sex",
    9: "Patient Alias",
    10: "Race",
    11: "Patient Address",
    12: "County Code",
    13: "Phone Number (Home)",
    14: "Phone Number (Business)",
    15: "Primary Language",
    16: "Marital Status",
    17: "Religion",
    18: "Patient Account Number",
    19: "SSN Number",
    20: "Driver's License Number",
    21: "Mother's Identifier",
    22: "Ethnic Group",
    23: "Birth Place",
    24: "Multiple Birth Indicator",
    25: "Birth Order",
    26: "Citizenship",
    27: "Veterans Military Status",
    28: "Nationality",
    29: "Patient Death Date/Time",
    30: "Patient Death Indicator",
}

PV1_FIELDS = {
    1: "Set ID",
    2: "Patient Class",
    3: "Assigned Patient Location",
    4: "Admission Type",
    5: "Preadmit Number",
    6: "Prior Patient Location",
    7: "Attending Doctor",
    8: "Referring Doctor",
    9: "Consulting Doctor",
    10: "Hospital Service",
    11: "Temporary Location",
    12: "Preadmit Test Indicator",
    13: "Re-admission Indicator",
    14: "Admit Source",
    15: "Ambulatory Status",
    16: "VIP Indicator",
    17: "Admitting Doctor",
    18: "Patient Type",
    19: "Visit Number",
    20: "Financial Class",
    44: "Admit Date/Time",
    45: "Discharge Date/Time",
}

OBX_FIELDS = {
    1: "Set ID",
    2: "Value Type",
    3: "Observation Identifier",
    4: "Observation Sub-ID",
    5: "Observation Value",
    6: "Units",
    7: "References Range",
    8: "Abnormal Flags",
    9: "Probability",
    10: "Nature of Abnormal Test",
    11: "Observation Result Status",
    12: "Effective Date of Reference Range",
    13: "User Defined Access Checks",
    14: "Date/Time of Observation",
    15: "Producer's Reference",
    16: "Responsible Observer",
    17: "Observation Method",
}

OBR_FIELDS = {
    1: "Set ID",
    2: "Placer Order Number",
    3: "Filler Order Number",
    4: "Universal Service Identifier",
    5: "Priority",
    6: "Requested Date/Time",
    7: "Observation Date/Time",
    8: "Observation End Date/Time",
    9: "Collection Volume",
    10: "Collector Identifier",
    11: "Specimen Action Code",
    12: "Danger Code",
    13: "Relevant Clinical Info",
    14: "Specimen Received Date/Time",
    15: "Specimen Source",
    16: "Ordering Provider",
    17: "Order Callback Phone Number",
    18: "Placer Field 1",
    19: "Placer Field 2",
    20: "Filler Field 1",
    21: "Filler Field 2",
    22: "Results Reported/Status Changed Date/Time",
    23: "Charge to Practice",
    24: "Diagnostic Service Section ID",
    25: "Result Status",
}

SEGMENT_FIELD_DESCRIPTIONS = {
    "MSH": MSH_FIELDS,
    "PID": PID_FIELDS,
    "PV1": PV1_FIELDS,
    "OBX": OBX_FIELDS,
    "OBR": OBR_FIELDS,
}

SAMPLE_HL7 = """MSH|^~\\&|ELI^93005||EMR^93005|Receiving FacID|20260520160615||ORU^R01|226052016061500654|P|2.5|||AL|NE|USA
PID|1||123456^^^MRN||DOE^JOHN^A||19800101|M|||123 MAIN ST^^SPRINGFIELD^IL^62701^USA||555-555-1234||EN|S|||SSN123456789
PV1|1|I|ICU^101^A||||1234^SMITH^JOHN^A|||SUR|||||||1234^SMITH^JOHN^A|IP||||||||||||||||||||||ADM|20231015080000
OBR|1|ORD001|FILL001|85025^CBC^L|||20231015100000|||||||20231015110000||1234^SMITH^JOHN^A|||||20231015115000||HM|F
OBX|1|NM|6690-2^WBC^LN||7.5|10*3/uL|4.5-11.0|N|||F|||20231015115000
OBX|2|NM|789-8^RBC^LN||4.8|10*6/uL|4.5-5.5|N|||F|||20231015115000
OBX|3|NM|718-7^HGB^LN||14.5|g/dL|13.5-17.5|N|||F|||20231015115000
OBX|4|NM|4544-3^HCT^LN||43.2|%|41.0-53.0|N|||F|||20231015115000
NTE|1||Patient fasting for 8 hours prior to collection"""


# ─────────────────────────────────────────────────────────────────────────────
# HL7 Parser  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
class HL7Parser:
    def __init__(self):
        self.field_sep = '|'
        self.component_sep = '^'
        self.repetition_sep = '~'
        self.escape_char = '\\'
        self.subcomponent_sep = '&'

    def parse(self, hl7_text):
        hl7_text = hl7_text.strip().replace('\r\n', '\r').replace('\n', '\r')
        segments = []
        for line in hl7_text.split('\r'):
            line = line.strip()
            if not line:
                continue
            seg_name = line[:3]
            if seg_name == 'MSH':
                if len(line) >= 4:
                    self.field_sep = line[3]
                if len(line) >= 8:
                    self.component_sep = line[4]
                    self.repetition_sep = line[5]
                    self.escape_char = line[6]
                    self.subcomponent_sep = line[7]
            if seg_name == 'MSH':
                raw_fields = line.split(self.field_sep)
                parsed_fields = []
                parsed_fields.append([[[raw_fields[0]]]])          # idx 0: "MSH"
                parsed_fields.append([[[self.field_sep]]])         # idx 1: MSH-1 = "|"
                enc = raw_fields[1] if len(raw_fields) > 1 else ''
                parsed_fields.append([[[enc]]])                    # idx 2: MSH-2 = "^~\&"
                for field in raw_fields[2:]:
                    reps = field.split(self.repetition_sep)
                    parsed_reps = []
                    for rep in reps:
                        comps = rep.split(self.component_sep)
                        parsed_comps = [comp.split(self.subcomponent_sep) for comp in comps]
                        parsed_reps.append(parsed_comps)
                    parsed_fields.append(parsed_reps)
                segments.append({'name': seg_name, 'raw': line, 'fields': parsed_fields})
            else:
                fields = line.split(self.field_sep)
                parsed_fields = []
                for field in fields:
                    reps = field.split(self.repetition_sep)
                    parsed_reps = []
                    for rep in reps:
                        comps = rep.split(self.component_sep)
                        parsed_comps = [comp.split(self.subcomponent_sep) for comp in comps]
                        parsed_reps.append(parsed_comps)
                    parsed_fields.append(parsed_reps)
                segments.append({'name': seg_name, 'raw': line, 'fields': parsed_fields})
        return segments

    def get_field_desc(self, seg_name, field_idx):
        descs = SEGMENT_FIELD_DESCRIPTIONS.get(seg_name, {})
        return descs.get(field_idx, f"Field {field_idx}")

    def rebuild(self, segments):
        return '\r\n'.join(seg['raw'] for seg in segments)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
class HL7ViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HL7 Viewer & Editor")
        self.root.geometry("1400x860")
        self.root.minsize(1000, 640)
        self.parser = HL7Parser()
        self.parsed_segments = []
        self.selected_segment_idx = None
        self.selected_field_idx = None
        self._pdf_temp_files = []      # track temp PDF files for cleanup
        self._setup_theme()
        self._build_ui()
        self._load_sample()

    # ── theme ────────────────────────────────────────────────────────────────
    def _setup_theme(self):
        self.colors = {
            'bg': '#1E2127',
            'panel': '#252A33',
            'panel2': '#2C3140',
            'accent': '#4D9EFF',
            'accent2': '#56D364',
            'accent3': '#FF7B50',
            'text': '#E6EDF3',
            'text_muted': '#8B949E',
            'border': '#30363D',
            'seg_header': '#2D333B',
            'field_row': '#21262D',
            'field_row_alt': '#1C2128',
            'highlight': '#3D4F6E',
            'error': '#F85149',
            'warning': '#D29922',
            'success': '#3FB950',
            'diff_add': '#1a3a2a',
            'diff_del': '#3a1a1a',
            'diff_chg': '#2a2a1a',
        }
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Panel.TFrame', background=self.colors['panel'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Consolas', 10))
        style.configure('Header.TLabel', background=self.colors['panel'], foreground=self.colors['text'],
                        font=('Segoe UI', 11, 'bold'))
        style.configure('Accent.TLabel', background=self.colors['panel'], foreground=self.colors['accent'],
                        font=('Consolas', 10, 'bold'))
        style.configure('Muted.TLabel', background=self.colors['panel'], foreground=self.colors['text_muted'],
                        font=('Segoe UI', 9))
        style.configure('TButton', background=self.colors['panel2'], foreground=self.colors['text'],
                        borderwidth=1, relief='flat', font=('Segoe UI', 9), padding=(8, 4))
        style.map('TButton',
                  background=[('active', self.colors['highlight']), ('pressed', self.colors['accent'])],
                  foreground=[('active', self.colors['text'])])
        style.configure('Accent.TButton', background=self.colors['accent'], foreground='white',
                        font=('Segoe UI', 9, 'bold'), padding=(10, 5))
        style.map('Accent.TButton',
                  background=[('active', '#3A8AEF'), ('pressed', '#2A7ADF')])
        style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['panel'], foreground=self.colors['text_muted'],
                        padding=(12, 6), font=('Segoe UI', 9))
        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['panel2'])],
                  foreground=[('selected', self.colors['accent'])])
        style.configure('Treeview', background=self.colors['field_row'], foreground=self.colors['text'],
                        fieldbackground=self.colors['field_row'], borderwidth=0,
                        font=('Consolas', 10), rowheight=26)
        style.configure('Treeview.Heading', background=self.colors['seg_header'],
                        foreground=self.colors['text_muted'], font=('Segoe UI', 9, 'bold'), relief='flat')
        style.map('Treeview', background=[('selected', self.colors['highlight'])],
                  foreground=[('selected', self.colors['text'])])
        style.configure('TScrollbar', background=self.colors['panel'], troughcolor=self.colors['bg'],
                        borderwidth=0, arrowcolor=self.colors['text_muted'])
        style.configure('TPanedwindow', background=self.colors['bg'])

    # ── build UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.configure(bg=self.colors['bg'])

        # ── top bar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=self.colors['panel'], height=50)
        topbar.pack(fill='x', side='top')
        topbar.pack_propagate(False)
        tk.Label(topbar, text="⚕ HL7 Viewer & Editor",
                 bg=self.colors['panel'], fg=self.colors['accent'],
                 font=('Segoe UI', 13, 'bold'), padx=16).pack(side='left', pady=8)
        tk.Label(topbar, text="v2.0 · Free & Open Source",
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(side='left', pady=8)
        btn_frame = tk.Frame(topbar, bg=self.colors['panel'])
        btn_frame.pack(side='right', padx=12, pady=8)
        for txt, cmd in [("📂 Open", self._open_file), ("💾 Save", self._save_file),
                         ("📋 Copy", self._copy_output), ("🗑 Clear", self._clear),
                         ("💡 Sample", self._load_sample)]:
            ttk.Button(btn_frame, text=txt, command=cmd, style='TButton').pack(side='left', padx=2)

        # ── status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready · Paste HL7 message to begin")
        statusbar = tk.Frame(self.root, bg=self.colors['panel'], height=26)
        statusbar.pack(fill='x', side='bottom')
        statusbar.pack_propagate(False)
        tk.Label(statusbar, textvariable=self.status_var,
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9), padx=12).pack(side='left', pady=3)
        self.msg_type_lbl = tk.Label(statusbar, text="",
                                     bg=self.colors['panel'], fg=self.colors['accent2'],
                                     font=('Segoe UI', 9, 'bold'), padx=12)
        self.msg_type_lbl.pack(side='right', pady=3)

        # ── main tab notebook (top-level tabs) ───────────────────────────────
        self.main_nb = ttk.Notebook(self.root)
        self.main_nb.pack(fill='both', expand=True)

        # Tab 1: Viewer/Editor
        viewer_frame = ttk.Frame(self.main_nb)
        self.main_nb.add(viewer_frame, text="  🔬 Viewer & Editor  ")

        # Tab 2: Compare
        compare_frame = ttk.Frame(self.main_nb)
        self.main_nb.add(compare_frame, text="  ⚖ Compare  ")

        # Tab 3: Base64 / PDF
        b64_frame = ttk.Frame(self.main_nb)
        self.main_nb.add(b64_frame, text="  📄 Base64 / PDF Viewer  ")

        # Tab 4: Statistics
        stats_frame = ttk.Frame(self.main_nb)
        self.main_nb.add(stats_frame, text="  📊 Statistics  ")

        self._build_viewer_tab(viewer_frame)
        self._build_compare_tab(compare_frame)
        self._build_b64_tab(b64_frame)
        self._build_stats_tab(stats_frame)

    # =========================================================================
    # TAB 1 — Viewer / Editor  (original layout preserved exactly)
    # =========================================================================
    def _build_viewer_tab(self, parent):
        main_pane = ttk.PanedWindow(parent, orient='horizontal')
        main_pane.pack(fill='both', expand=True)

        # ── left: input/output notebook ──────────────────────────────────────
        left_frame = ttk.Frame(main_pane, style='Panel.TFrame')
        main_pane.add(left_frame, weight=45)

        nb = ttk.Notebook(left_frame)
        nb.pack(fill='both', expand=True)

        # Input tab
        input_frame = ttk.Frame(nb)
        nb.add(input_frame, text="  📥 Input  ")

        # ── NEW: segment filter bar ──────────────────────────────────────────
        filter_bar = tk.Frame(input_frame, bg=self.colors['panel2'])
        filter_bar.pack(fill='x', padx=0, pady=0)
        tk.Label(filter_bar, text=" 🔍 Filter:", bg=self.colors['panel2'],
                 fg=self.colors['text_muted'], font=('Segoe UI', 9)).pack(side='left', pady=4)
        self.filter_var = tk.StringVar()
        filter_entry = tk.Entry(filter_bar, textvariable=self.filter_var,
                                bg=self.colors['panel2'], fg=self.colors['text'],
                                insertbackground=self.colors['accent'],
                                font=('Consolas', 10), relief='flat', width=18)
        filter_entry.pack(side='left', padx=4, pady=4, ipady=2)
        filter_entry.bind('<KeyRelease>', lambda e: self._apply_filter())
        tk.Label(filter_bar, text="Segment:", bg=self.colors['panel2'],
                 fg=self.colors['text_muted'], font=('Segoe UI', 9)).pack(side='left', pady=4, padx=(8,2))
        self.seg_filter_var = tk.StringVar(value="ALL")
        self.seg_filter_cb = ttk.Combobox(filter_bar, textvariable=self.seg_filter_var,
                                          state='readonly', width=8, font=('Segoe UI', 9))
        self.seg_filter_cb['values'] = ['ALL']
        self.seg_filter_cb.pack(side='left', pady=4)
        self.seg_filter_cb.bind('<<ComboboxSelected>>', lambda e: self._apply_filter())
        ttk.Button(filter_bar, text="✖", command=self._clear_filter, style='TButton',
                   width=2).pack(side='left', padx=4)

        tk.Label(input_frame, text="Paste HL7 Message Here:",
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9), pady=4).pack(anchor='w', padx=8)
        inp_scroll = tk.Scrollbar(input_frame, bg=self.colors['panel'])
        inp_scroll.pack(side='right', fill='y')
        self.input_text = tk.Text(input_frame, bg=self.colors['bg'], fg=self.colors['text'],
                                  insertbackground=self.colors['accent'],
                                  font=('Consolas', 10), wrap='none', relief='flat',
                                  undo=True, yscrollcommand=inp_scroll.set,
                                  selectbackground=self.colors['highlight'],
                                  padx=8, pady=6)
        self.input_text.pack(fill='both', expand=True)
        inp_scroll.config(command=self.input_text.yview)
        self.input_text.bind('<KeyRelease>', self._on_input_change)
        self.input_text.bind('<Control-Return>', lambda e: self._parse_message())
        inp_xscroll = tk.Scrollbar(input_frame, orient='horizontal', bg=self.colors['panel'])
        inp_xscroll.pack(fill='x')
        self.input_text.configure(xscrollcommand=inp_xscroll.set)
        inp_xscroll.config(command=self.input_text.xview)
        ttk.Button(input_frame, text="▶ Parse Message  (Ctrl+Enter)",
                   command=self._parse_message, style='Accent.TButton').pack(fill='x', padx=8, pady=6)

        # Output tab
        output_frame = ttk.Frame(nb)
        nb.add(output_frame, text="  📤 Output  ")
        tk.Label(output_frame, text="Rebuilt / Edited HL7:",
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9), pady=4).pack(anchor='w', padx=8)
        out_scroll = tk.Scrollbar(output_frame, bg=self.colors['panel'])
        out_scroll.pack(side='right', fill='y')
        self.output_text = tk.Text(output_frame, bg=self.colors['bg'], fg=self.colors['accent2'],
                                   insertbackground=self.colors['accent'],
                                   font=('Consolas', 10), wrap='none', relief='flat',
                                   yscrollcommand=out_scroll.set,
                                   selectbackground=self.colors['highlight'],
                                   padx=8, pady=6)
        self.output_text.pack(fill='both', expand=True)
        out_scroll.config(command=self.output_text.yview)
        out_xscroll = tk.Scrollbar(output_frame, orient='horizontal', bg=self.colors['panel'])
        out_xscroll.pack(fill='x')
        self.output_text.configure(xscrollcommand=out_xscroll.set)
        out_xscroll.config(command=self.output_text.xview)
        self.nb = nb

        # ── right: segment tree + field editor ───────────────────────────────
        right_pane = ttk.PanedWindow(main_pane, orient='vertical')
        main_pane.add(right_pane, weight=55)

        # Segment tree
        seg_frame = ttk.Frame(right_pane, style='Panel.TFrame')
        right_pane.add(seg_frame, weight=50)
        seg_hdr = tk.Frame(seg_frame, bg=self.colors['seg_header'], height=32)
        seg_hdr.pack(fill='x')
        seg_hdr.pack_propagate(False)
        tk.Label(seg_hdr, text="  🔍 Segment Browser",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=6)
        self.seg_count_lbl = tk.Label(seg_hdr, text="",
                                      bg=self.colors['seg_header'], fg=self.colors['text_muted'],
                                      font=('Segoe UI', 9))
        self.seg_count_lbl.pack(side='right', padx=10, pady=6)
        seg_tree_frame = ttk.Frame(seg_frame)
        seg_tree_frame.pack(fill='both', expand=True)
        seg_scroll_y = ttk.Scrollbar(seg_tree_frame, orient='vertical')
        seg_scroll_y.pack(side='right', fill='y')
        seg_scroll_x = ttk.Scrollbar(seg_tree_frame, orient='horizontal')
        seg_scroll_x.pack(side='bottom', fill='x')
        self.seg_tree = ttk.Treeview(seg_tree_frame, columns=('field', 'value', 'desc'),
                                     show='tree headings',
                                     yscrollcommand=seg_scroll_y.set,
                                     xscrollcommand=seg_scroll_x.set)
        self.seg_tree.heading('#0', text='Segment')
        self.seg_tree.heading('field', text='Field #')
        self.seg_tree.heading('value', text='Value')
        self.seg_tree.heading('desc', text='Description')
        self.seg_tree.column('#0', width=120, minwidth=80)
        self.seg_tree.column('field', width=70, minwidth=50)
        self.seg_tree.column('value', width=260, minwidth=100)
        self.seg_tree.column('desc', width=200, minwidth=100)
        self.seg_tree.pack(fill='both', expand=True)
        seg_scroll_y.config(command=self.seg_tree.yview)
        seg_scroll_x.config(command=self.seg_tree.xview)
        self.seg_tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.seg_tree.tag_configure('segment', font=('Segoe UI', 10, 'bold'),
                                    foreground=self.colors['accent'])
        self.seg_tree.tag_configure('field', foreground=self.colors['text'])
        self.seg_tree.tag_configure('field_alt', foreground=self.colors['text'],
                                    background=self.colors['field_row_alt'])
        self.seg_tree.tag_configure('empty', foreground=self.colors['text_muted'])

        # Field editor
        edit_frame = ttk.Frame(right_pane, style='Panel.TFrame')
        right_pane.add(edit_frame, weight=50)
        edit_hdr = tk.Frame(edit_frame, bg=self.colors['seg_header'], height=32)
        edit_hdr.pack(fill='x')
        edit_hdr.pack_propagate(False)
        tk.Label(edit_hdr, text="  ✏ Field Editor & Component View",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=6)

        # ── NEW: field search bar inside editor header ──────────────────────
        tk.Label(edit_hdr, text="  Search field:",
                 bg=self.colors['seg_header'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(side='left', pady=6)
        self.field_search_var = tk.StringVar()
        fsearch = tk.Entry(edit_hdr, textvariable=self.field_search_var,
                           bg=self.colors['panel2'], fg=self.colors['text'],
                           insertbackground=self.colors['accent'],
                           font=('Consolas', 9), relief='flat', width=18)
        fsearch.pack(side='left', pady=6, padx=4, ipady=2)
        fsearch.bind('<Return>', self._field_search_next)
        ttk.Button(edit_hdr, text="▶", command=self._field_search_next,
                   style='TButton', width=2).pack(side='left', pady=6)

        edit_content = tk.Frame(edit_frame, bg=self.colors['panel'])
        edit_content.pack(fill='both', expand=True, padx=8, pady=6)
        info_row = tk.Frame(edit_content, bg=self.colors['panel'])
        info_row.pack(fill='x', pady=(0, 6))
        self.field_path_var = tk.StringVar(value="Select a field from the segment browser →")
        tk.Label(info_row, textvariable=self.field_path_var,
                 bg=self.colors['panel'], fg=self.colors['accent'],
                 font=('Consolas', 10, 'bold')).pack(side='left')
        self.field_desc_var = tk.StringVar(value="")
        tk.Label(info_row, textvariable=self.field_desc_var,
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(side='left', padx=10)
        raw_lbl_row = tk.Frame(edit_content, bg=self.colors['panel'])
        raw_lbl_row.pack(fill='x')
        tk.Label(raw_lbl_row, text="Raw Field Value:",
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(side='left')
        self.field_editor = tk.Entry(edit_content, bg=self.colors['panel2'],
                                     fg=self.colors['text'],
                                     insertbackground=self.colors['accent'],
                                     font=('Consolas', 11), relief='flat',
                                     selectbackground=self.colors['highlight'])
        self.field_editor.pack(fill='x', pady=4, ipady=6)
        self.field_editor.bind('<Return>', self._apply_edit)
        btn_row = tk.Frame(edit_content, bg=self.colors['panel'])
        btn_row.pack(fill='x', pady=(0, 8))
        ttk.Button(btn_row, text="✔ Apply Change (Enter)",
                   command=self._apply_edit, style='Accent.TButton').pack(side='left', padx=(0, 6))
        ttk.Button(btn_row, text="✖ Reset Field",
                   command=self._reset_field, style='TButton').pack(side='left')
        # ── NEW: View as PDF button (shown when field has base64 data) ───────
        self.view_pdf_btn = ttk.Button(btn_row, text="📄 View as PDF",
                                       command=self._view_selected_field_as_pdf,
                                       style='TButton')
        self.view_pdf_btn.pack(side='left', padx=8)
        self.view_pdf_btn.state(['disabled'])

        tk.Label(edit_content, text="Component Breakdown  (^ delimited):",
                 bg=self.colors['panel'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(4, 2))
        comp_scroll = tk.Scrollbar(edit_content)
        comp_scroll.pack(side='right', fill='y')
        self.comp_tree = ttk.Treeview(edit_content,
                                      columns=('comp_idx', 'value', 'subcomps'),
                                      show='headings', height=6,
                                      yscrollcommand=comp_scroll.set)
        self.comp_tree.heading('comp_idx', text='Component #')
        self.comp_tree.heading('value', text='Value')
        self.comp_tree.heading('subcomps', text='Sub-components (&)')
        self.comp_tree.column('comp_idx', width=100, minwidth=70)
        self.comp_tree.column('value', width=250, minwidth=100)
        self.comp_tree.column('subcomps', width=200, minwidth=100)
        self.comp_tree.pack(fill='both', expand=True)
        comp_scroll.config(command=self.comp_tree.yview)
        self.comp_tree.tag_configure('comp', foreground=self.colors['text'])
        self.comp_tree.tag_configure('comp_alt', foreground=self.colors['text'],
                                     background=self.colors['field_row_alt'])
        self.rep_var = tk.StringVar(value="")
        tk.Label(edit_content, textvariable=self.rep_var,
                 bg=self.colors['panel'], fg=self.colors['warning'],
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(4, 0))

    # =========================================================================
    # TAB 2 — Compare
    # =========================================================================
    def _build_compare_tab(self, parent):
        parent.configure(style='Panel.TFrame')
        hdr = tk.Frame(parent, bg=self.colors['seg_header'], height=36)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  ⚖ HL7 Message Compare — paste two messages to diff them",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=8)
        ttk.Button(hdr, text="▶ Compare", command=self._do_compare,
                   style='Accent.TButton').pack(side='right', padx=10, pady=6)
        ttk.Button(hdr, text="🗑 Clear", command=self._clear_compare,
                   style='TButton').pack(side='right', padx=4, pady=6)
        ttk.Button(hdr, text="⬆ Load from Editor (A)", command=self._load_compare_from_editor,
                   style='TButton').pack(side='right', padx=4, pady=6)

        # Top: two input panes side by side
        input_pane = ttk.PanedWindow(parent, orient='horizontal')
        input_pane.pack(fill='both', expand=True)

        # Message A
        a_frame = tk.Frame(input_pane, bg=self.colors['panel'])
        input_pane.add(a_frame, weight=1)
        tk.Label(a_frame, text="  📨 Message A  (original)",
                 bg=self.colors['seg_header'], fg=self.colors['accent'],
                 font=('Segoe UI', 9, 'bold')).pack(fill='x')
        a_scroll = tk.Scrollbar(a_frame)
        a_scroll.pack(side='right', fill='y')
        self.cmp_a_text = tk.Text(a_frame, bg=self.colors['bg'], fg=self.colors['text'],
                                  font=('Consolas', 10), wrap='none', relief='flat',
                                  yscrollcommand=a_scroll.set,
                                  insertbackground=self.colors['accent'],
                                  selectbackground=self.colors['highlight'],
                                  padx=8, pady=6, height=10)
        self.cmp_a_text.pack(fill='both', expand=True)
        a_scroll.config(command=self.cmp_a_text.yview)

        # Message B
        b_frame = tk.Frame(input_pane, bg=self.colors['panel'])
        input_pane.add(b_frame, weight=1)
        tk.Label(b_frame, text="  📩 Message B  (modified)",
                 bg=self.colors['seg_header'], fg=self.colors['accent2'],
                 font=('Segoe UI', 9, 'bold')).pack(fill='x')
        b_scroll = tk.Scrollbar(b_frame)
        b_scroll.pack(side='right', fill='y')
        self.cmp_b_text = tk.Text(b_frame, bg=self.colors['bg'], fg=self.colors['text'],
                                  font=('Consolas', 10), wrap='none', relief='flat',
                                  yscrollcommand=b_scroll.set,
                                  insertbackground=self.colors['accent'],
                                  selectbackground=self.colors['highlight'],
                                  padx=8, pady=6, height=10)
        self.cmp_b_text.pack(fill='both', expand=True)
        b_scroll.config(command=self.cmp_b_text.yview)

        # Result area
        res_hdr = tk.Frame(parent, bg=self.colors['seg_header'], height=28)
        res_hdr.pack(fill='x')
        res_hdr.pack_propagate(False)
        tk.Label(res_hdr, text="  Diff Result",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 9, 'bold')).pack(side='left', pady=5)
        self.cmp_summary_var = tk.StringVar(value="")
        tk.Label(res_hdr, textvariable=self.cmp_summary_var,
                 bg=self.colors['seg_header'], fg=self.colors['warning'],
                 font=('Segoe UI', 9)).pack(side='right', padx=10, pady=5)

        res_frame = tk.Frame(parent, bg=self.colors['panel'])
        res_frame.pack(fill='both', expand=True)
        res_scroll_y = tk.Scrollbar(res_frame)
        res_scroll_y.pack(side='right', fill='y')
        res_scroll_x = tk.Scrollbar(res_frame, orient='horizontal')
        res_scroll_x.pack(side='bottom', fill='x')
        self.cmp_result = tk.Text(res_frame, bg=self.colors['bg'], fg=self.colors['text'],
                                  font=('Consolas', 10), wrap='none', relief='flat',
                                  state='disabled',
                                  yscrollcommand=res_scroll_y.set,
                                  xscrollcommand=res_scroll_x.set,
                                  padx=8, pady=6)
        self.cmp_result.pack(fill='both', expand=True)
        res_scroll_y.config(command=self.cmp_result.yview)
        res_scroll_x.config(command=self.cmp_result.xview)

        # Legend
        leg = tk.Frame(parent, bg=self.colors['panel2'])
        leg.pack(fill='x')
        for color, label in [(self.colors['diff_add'], '  ● Added  '),
                              (self.colors['diff_del'], '  ● Removed  '),
                              (self.colors['diff_chg'], '  ● Changed  ')]:
            tk.Label(leg, text=label, bg=color, fg=self.colors['text'],
                     font=('Segoe UI', 9)).pack(side='left', padx=4, pady=3)

        self.cmp_result.tag_configure('add', background=self.colors['diff_add'],
                                      foreground=self.colors['accent2'])
        self.cmp_result.tag_configure('del', background=self.colors['diff_del'],
                                      foreground=self.colors['error'])
        self.cmp_result.tag_configure('chg', background=self.colors['diff_chg'],
                                      foreground=self.colors['warning'])
        self.cmp_result.tag_configure('same', foreground=self.colors['text_muted'])
        self.cmp_result.tag_configure('hdr', foreground=self.colors['text_muted'],
                                      font=('Segoe UI', 8))

    # =========================================================================
    # TAB 3 — Base64 / PDF Viewer
    # =========================================================================
    def _build_b64_tab(self, parent):
        hdr = tk.Frame(parent, bg=self.colors['seg_header'], height=36)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📄 Base64 Decoder & PDF Viewer",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=8)

        body = ttk.PanedWindow(parent, orient='horizontal')
        body.pack(fill='both', expand=True)

        # Left: input
        left = tk.Frame(body, bg=self.colors['panel'])
        body.add(left, weight=45)
        tk.Label(left, text="  Paste Base64-encoded data here:",
                 bg=self.colors['panel2'], fg=self.colors['text_muted'],
                 font=('Segoe UI', 9)).pack(fill='x')
        b64_scroll = tk.Scrollbar(left)
        b64_scroll.pack(side='right', fill='y')
        self.b64_input = tk.Text(left, bg=self.colors['bg'], fg=self.colors['text'],
                                 font=('Consolas', 9), wrap='none', relief='flat',
                                 yscrollcommand=b64_scroll.set,
                                 insertbackground=self.colors['accent'],
                                 selectbackground=self.colors['highlight'],
                                 padx=8, pady=6)
        self.b64_input.pack(fill='both', expand=True)
        b64_scroll.config(command=self.b64_input.yview)

        # Buttons
        bbar = tk.Frame(left, bg=self.colors['panel'])
        bbar.pack(fill='x', padx=6, pady=6)
        ttk.Button(bbar, text="🔓 Decode & Preview", command=self._decode_b64,
                   style='Accent.TButton').pack(side='left', padx=2)
        ttk.Button(bbar, text="📄 Open PDF in Browser", command=self._open_pdf_browser,
                   style='TButton').pack(side='left', padx=2)
        ttk.Button(bbar, text="💾 Save Decoded File", command=self._save_decoded,
                   style='TButton').pack(side='left', padx=2)
        ttk.Button(bbar, text="🗑 Clear", command=self._clear_b64,
                   style='TButton').pack(side='left', padx=2)
        ttk.Button(bbar, text="⬆ Import from OBX-5",
                   command=self._import_obx5_b64, style='TButton').pack(side='left', padx=2)

        # Right: decoded output
        right = tk.Frame(body, bg=self.colors['panel'])
        body.add(right, weight=55)
        self.b64_info_var = tk.StringVar(value="Decoded output will appear here")
        tk.Label(right, textvariable=self.b64_info_var,
                 bg=self.colors['panel2'], fg=self.colors['accent'],
                 font=('Segoe UI', 9, 'bold')).pack(fill='x')
        out_scroll = tk.Scrollbar(right)
        out_scroll.pack(side='right', fill='y')
        self.b64_output = tk.Text(right, bg=self.colors['bg'], fg=self.colors['accent2'],
                                  font=('Consolas', 9), wrap='word', relief='flat',
                                  state='disabled',
                                  yscrollcommand=out_scroll.set,
                                  padx=8, pady=6)
        self.b64_output.pack(fill='both', expand=True)
        out_scroll.config(command=self.b64_output.yview)

        self._b64_decoded_bytes = None  # store last decoded bytes

    # =========================================================================
    # TAB 4 — Statistics  (HL7 Inspector-style)
    # =========================================================================
    def _build_stats_tab(self, parent):
        hdr = tk.Frame(parent, bg=self.colors['seg_header'], height=36)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📊 Message Statistics & Summary",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=8)
        ttk.Button(hdr, text="🔄 Refresh", command=self._refresh_stats,
                   style='TButton').pack(side='right', padx=10, pady=6)

        body = tk.Frame(parent, bg=self.colors['bg'])
        body.pack(fill='both', expand=True, padx=12, pady=10)

        # Top summary cards row
        self.stats_cards_frame = tk.Frame(body, bg=self.colors['bg'])
        self.stats_cards_frame.pack(fill='x', pady=(0, 10))

        # Segment frequency table
        tk.Label(body, text="Segment Frequency",
                 bg=self.colors['bg'], fg=self.colors['accent'],
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w')

        tbl_frame = tk.Frame(body, bg=self.colors['panel'])
        tbl_frame.pack(fill='both', expand=True, pady=(4, 0))
        tbl_scroll = ttk.Scrollbar(tbl_frame)
        tbl_scroll.pack(side='right', fill='y')
        self.stats_tree = ttk.Treeview(tbl_frame,
                                       columns=('count', 'desc', 'obx_types'),
                                       show='headings',
                                       yscrollcommand=tbl_scroll.set)
        self.stats_tree.heading('count', text='Count')
        self.stats_tree.heading('desc', text='Description')
        self.stats_tree.heading('obx_types', text='OBX Value Types / Notes')
        self.stats_tree.column('count', width=60, minwidth=50, anchor='center')
        self.stats_tree.column('desc', width=220, minwidth=120)
        self.stats_tree.column('obx_types', width=340, minwidth=120)
        self.stats_tree.pack(fill='both', expand=True)
        tbl_scroll.config(command=self.stats_tree.yview)

    # =========================================================================
    # Viewer logic  (original, unchanged)
    # =========================================================================
    def _on_input_change(self, event=None):
        if self.input_text.get('1.0', 'end').strip():
            self.status_var.set("Press Ctrl+Enter or click 'Parse Message' to analyze")

    def _parse_message(self, text=None):
        if text is None:
            text = self.input_text.get('1.0', 'end').strip()
        if not text:
            self.status_var.set("⚠ No HL7 content to parse")
            return
        try:
            self.parsed_segments = self.parser.parse(text)
            self._populate_tree()
            self._update_output()
            self._detect_msg_type()
            self._update_seg_filter_choices()
            self.status_var.set(
                f"✔ Parsed {len(self.parsed_segments)} segments successfully · "
                f"{datetime.now().strftime('%H:%M:%S')}")
            self.nb.select(0)
        except Exception as e:
            self.status_var.set(f"✖ Parse error: {str(e)}")
            messagebox.showerror("Parse Error", str(e))

    def _detect_msg_type(self):
        for seg in self.parsed_segments:
            if seg['name'] == 'MSH':
                fields = seg['fields']
                try:
                    if len(fields) > 10:
                        msg_type = fields[10][0][0][0] if fields[10] else ''
                        trigger  = fields[10][0][1][0] if (fields[10] and len(fields[10][0]) > 1) else ''
                        version  = fields[13][0][0][0] if (len(fields) > 13 and fields[13]) else ''
                        if msg_type:
                            self.msg_type_lbl.config(text=f"  {msg_type}^{trigger}  |  HL7 v{version}  ")
                except Exception:
                    pass

    def _populate_tree(self, segments=None):
        for item in self.seg_tree.get_children():
            self.seg_tree.delete(item)
        segs = segments if segments is not None else self.parsed_segments
        self.seg_count_lbl.config(text=f"{len(segs)} segments")
        for s_idx, seg in enumerate(segs):
            seg_name = seg['name']
            seg_desc = HL7_SEGMENT_DESCRIPTIONS.get(seg_name, "Custom/Unknown Segment")
            seg_node = self.seg_tree.insert('', 'end',
                                            text=f"  {seg_name}",
                                            values=('', '', seg_desc),
                                            tags=('segment',),
                                            iid=f"seg_{s_idx}")
            for f_idx, field in enumerate(seg['fields']):
                if f_idx == 0:
                    continue
                raw_val = self._field_to_raw(field)
                desc = self.parser.get_field_desc(seg_name, f_idx)
                tag = 'field' if f_idx % 2 == 0 else 'field_alt'
                if not raw_val:
                    tag = 'empty'
                self.seg_tree.insert(seg_node, 'end',
                                     text=f"   {seg_name}",
                                     values=(f"[{f_idx}]",
                                             raw_val[:80] + ('…' if len(raw_val) > 80 else ''),
                                             desc),
                                     tags=(tag,),
                                     iid=f"seg_{s_idx}_fld_{f_idx}")
            self.seg_tree.item(seg_node, open=True)

    def _field_to_raw(self, field):
        reps = []
        for rep in field:
            comps = []
            for comp in rep:
                subs = self.parser.subcomponent_sep.join(comp)
                comps.append(subs)
            reps.append(self.parser.component_sep.join(comps))
        return self.parser.repetition_sep.join(reps)

    def _on_tree_select(self, event=None):
        sel = self.seg_tree.selection()
        if not sel:
            return
        iid = sel[0]
        parts = iid.split('_')
        if len(parts) == 2:
            s_idx = int(parts[1])
            seg = self.parsed_segments[s_idx]
            self.field_path_var.set(f"Segment: {seg['name']}  (row {s_idx + 1})")
            self.field_desc_var.set(HL7_SEGMENT_DESCRIPTIONS.get(seg['name'], ''))
            self.field_editor.delete(0, 'end')
            self.field_editor.insert(0, seg['raw'])
            self.selected_segment_idx = s_idx
            self.selected_field_idx = None
            self._clear_comp_tree()
            self.view_pdf_btn.state(['disabled'])
        elif len(parts) == 4:
            s_idx = int(parts[1])
            f_idx = int(parts[3])
            seg = self.parsed_segments[s_idx]
            field = seg['fields'][f_idx]
            raw_val = self._field_to_raw(field)
            desc = self.parser.get_field_desc(seg['name'], f_idx)
            self.field_path_var.set(f"{seg['name']}-{f_idx}")
            self.field_desc_var.set(f"→ {desc}")
            self.field_editor.delete(0, 'end')
            self.field_editor.insert(0, raw_val)
            self.selected_segment_idx = s_idx
            self.selected_field_idx = f_idx
            self._show_components(field)
            self.rep_var.set(f"⟳ {len(field)} repetition(s) — separated by ~" if len(field) > 1 else "")
            # Enable PDF button if field looks like base64
            if self._looks_like_base64(raw_val):
                self.view_pdf_btn.state(['!disabled'])
            else:
                self.view_pdf_btn.state(['disabled'])

    def _show_components(self, field):
        self._clear_comp_tree()
        for rep_idx, rep in enumerate(field):
            for comp_idx, comp in enumerate(rep):
                subcomp_str = ' & '.join(comp) if len(comp) > 1 else ''
                val = comp[0] if comp else ''
                tag = 'comp' if comp_idx % 2 == 0 else 'comp_alt'
                rep_str = f"Rep {rep_idx+1}, " if len(field) > 1 else ""
                self.comp_tree.insert('', 'end',
                                      values=(f"{rep_str}Comp {comp_idx+1}", val, subcomp_str),
                                      tags=(tag,))

    def _clear_comp_tree(self):
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)

    def _apply_edit(self, event=None):
        if self.selected_segment_idx is None:
            return
        new_val = self.field_editor.get()
        s_idx = self.selected_segment_idx
        seg = self.parsed_segments[s_idx]
        if self.selected_field_idx is None:
            seg['raw'] = new_val
            fields_raw = new_val.split(self.parser.field_sep)
            parsed_fields = []
            for f in fields_raw:
                reps = f.split(self.parser.repetition_sep)
                parsed_reps = []
                for rep in reps:
                    comps = rep.split(self.parser.component_sep)
                    parsed_comps = [comp.split(self.parser.subcomponent_sep) for comp in comps]
                    parsed_reps.append(parsed_comps)
                parsed_fields.append(parsed_reps)
            seg['fields'] = parsed_fields
        else:
            f_idx = self.selected_field_idx
            reps = new_val.split(self.parser.repetition_sep)
            parsed_reps = []
            for rep in reps:
                comps = rep.split(self.parser.component_sep)
                parsed_comps = [comp.split(self.parser.subcomponent_sep) for comp in comps]
                parsed_reps.append(parsed_comps)
            seg['fields'][f_idx] = parsed_reps
            if seg['name'] == 'MSH':
                msh_data = [self._field_to_raw(fld) for fld in seg['fields'][3:]]
                seg['raw'] = ('MSH' + self.parser.field_sep
                              + self.parser.component_sep + self.parser.repetition_sep
                              + self.parser.escape_char + self.parser.subcomponent_sep
                              + self.parser.field_sep + self.parser.field_sep.join(msh_data))
            else:
                seg['raw'] = self.parser.field_sep.join(self._field_to_raw(f) for f in seg['fields'])
        self._populate_tree()
        self._update_output()
        self.status_var.set(f"✔ Field updated · {datetime.now().strftime('%H:%M:%S')}")
        try:
            if self.selected_field_idx is not None:
                self.seg_tree.selection_set(f"seg_{s_idx}_fld_{self.selected_field_idx}")
            else:
                self.seg_tree.selection_set(f"seg_{s_idx}")
        except Exception:
            pass

    def _reset_field(self):
        sel = self.seg_tree.selection()
        if not sel:
            return
        iid = sel[0]
        parts = iid.split('_')
        if len(parts) == 4:
            s_idx, f_idx = int(parts[1]), int(parts[3])
            raw = self._field_to_raw(self.parsed_segments[s_idx]['fields'][f_idx])
            self.field_editor.delete(0, 'end')
            self.field_editor.insert(0, raw)

    def _update_output(self):
        result = '\r\n'.join(seg['raw'] for seg in self.parsed_segments)
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', result)
        self.nb.select(1)
        self.nb.select(0)

    def _copy_output(self):
        content = self.output_text.get('1.0', 'end').strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("✔ Output copied to clipboard")
        else:
            self.status_var.set("⚠ No output to copy")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open HL7 File",
            filetypes=[("HL7 Files", "*.hl7 *.txt *.msg"), ("All Files", "*.*")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                self.input_text.delete('1.0', 'end')
                self.input_text.insert('1.0', content)
                self._parse_message(content)
                self.status_var.set(f"✔ Opened: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")

    def _save_file(self):
        content = self.output_text.get('1.0', 'end').strip() or self.input_text.get('1.0', 'end').strip()
        if not content:
            messagebox.showwarning("Nothing to Save", "No HL7 content to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save HL7 File", defaultextension=".hl7",
            filetypes=[("HL7 Files", "*.hl7"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_var.set(f"✔ Saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    def _clear(self):
        self.input_text.delete('1.0', 'end')
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        for item in self.seg_tree.get_children():
            self.seg_tree.delete(item)
        self._clear_comp_tree()
        self.field_editor.delete(0, 'end')
        self.field_path_var.set("Select a field from the segment browser →")
        self.field_desc_var.set("")
        self.rep_var.set("")
        self.msg_type_lbl.config(text="")
        self.seg_count_lbl.config(text="")
        self.parsed_segments = []
        self.selected_segment_idx = None
        self.selected_field_idx = None
        self.filter_var.set("")
        self.seg_filter_var.set("ALL")
        self.seg_filter_cb['values'] = ['ALL']
        self.view_pdf_btn.state(['disabled'])
        self.status_var.set("Cleared · Paste HL7 message to begin")

    def _load_sample(self):
        self._clear()
        self.input_text.insert('1.0', SAMPLE_HL7)
        self._parse_message(SAMPLE_HL7)
        self.status_var.set("✔ Sample HL7 ORU^R01 message loaded")

    # =========================================================================
    # NEW: Segment filter / field search
    # =========================================================================
    def _update_seg_filter_choices(self):
        names = ['ALL'] + sorted(set(s['name'] for s in self.parsed_segments))
        self.seg_filter_cb['values'] = names
        self.seg_filter_var.set('ALL')

    def _apply_filter(self):
        text_filter = self.filter_var.get().strip().lower()
        seg_filter = self.seg_filter_var.get()
        if not self.parsed_segments:
            return
        filtered = []
        for seg in self.parsed_segments:
            if seg_filter != 'ALL' and seg['name'] != seg_filter:
                continue
            if text_filter and text_filter not in seg['raw'].lower():
                continue
            filtered.append(seg)
        self._populate_tree(filtered)
        self.seg_count_lbl.config(text=f"{len(filtered)} of {len(self.parsed_segments)} segments")

    def _clear_filter(self):
        self.filter_var.set("")
        self.seg_filter_var.set("ALL")
        self._populate_tree()
        self.seg_count_lbl.config(text=f"{len(self.parsed_segments)} segments")

    def _field_search_next(self, event=None):
        """Highlight the next field in the tree whose value contains the search term."""
        term = self.field_search_var.get().strip().lower()
        if not term or not self.parsed_segments:
            return
        # Collect all field iids in order
        all_iids = []
        for item in self.seg_tree.get_children():
            for child in self.seg_tree.get_children(item):
                all_iids.append(child)
        if not all_iids:
            return
        # Find current selection position
        sel = self.seg_tree.selection()
        start_idx = 0
        if sel:
            try:
                start_idx = all_iids.index(sel[0]) + 1
            except ValueError:
                start_idx = 0
        # Search from current+1, wrap around
        for i in range(len(all_iids)):
            idx = (start_idx + i) % len(all_iids)
            iid = all_iids[idx]
            vals = self.seg_tree.item(iid, 'values')
            if term in str(vals[1]).lower():
                self.seg_tree.selection_set(iid)
                self.seg_tree.see(iid)
                self._on_tree_select()
                self.status_var.set(f"🔍 Found '{term}' in {iid}")
                return
        self.status_var.set(f"⚠ '{term}' not found in any field value")

    # =========================================================================
    # NEW: Base64 / PDF logic
    # =========================================================================
    # ── Base64 / HL7 ED helpers (handles ALL real-world formats) ─────────────
    # Format A: ^AP^PDF^Base64^<data>           (encoding descriptor)
    # Format B: ^AP^PDF^Base 64 Encoding^<data> (verbose descriptor)
    # Format C: PDF^IMAGE^^^<data>               (BLANK encoding, data in last component)
    # Format D: <plain_base64>                   (no HL7 wrapper)

    def _clean_ctrl(self, s):
        """Remove all control chars and whitespace for validation."""
        import re
        return re.sub(r'[-\s]', '', s)

    def _is_base64_encoding(self, enc):
        """Check if encoding descriptor means base64 — handles all variants."""
        n = self._clean_ctrl(enc).upper()
        return 'BASE64' in n or n == 'B64'

    def _looks_like_b64_data(self, s, min_len=100):
        """Is this string likely base64 data (not a short descriptor)?"""
        import re
        if not s or len(s.strip()) < 4:
            return False
        clean = self._clean_ctrl(s)
        if len(clean) < min_len:
            return False
        return bool(re.match(r'^[A-Za-z0-9+/]+=*$', clean))

    def _extract_base64(self, raw):
        """
        Smart base64 extraction from ANY HL7 ED field format.
        Tries encoding descriptor first, then finds largest b64 component,
        then treats whole string as b64.
        """
        raw = raw.strip()
        parts = raw.split('^')

        # Strategy 1: encoding descriptor in parts[3] -> payload in parts[4]
        if len(parts) >= 5 and self._is_base64_encoding(parts[3]):
            payload = parts[4].strip()
            if payload and self._looks_like_b64_data(payload):
                return payload

        # Strategy 2: find the LARGEST component that looks like base64
        # (handles Format C: PDF^IMAGE^^^<data> with blank encoding)
        best = None
        best_len = 100
        for part in parts:
            p = part.strip()
            if self._looks_like_b64_data(p):
                clean_len = len(self._clean_ctrl(p))
                if clean_len > best_len:
                    best = p
                    best_len = clean_len
        if best:
            return best

        # Strategy 3: whole string is base64
        if self._looks_like_b64_data(raw):
            return raw

        return raw  # return as-is; decoder will fail gracefully

    def _looks_like_base64(self, s):
        """Returns True if raw field value appears to carry base64 data."""
        s = (s or '').strip()
        if not s or len(s) < 20:
            return False
        parts = s.split('^')
        # Check encoding descriptor
        if len(parts) >= 5 and self._is_base64_encoding(parts[3]):
            return True
        # Check if any component is large and looks like base64
        for part in parts:
            if self._looks_like_b64_data(part.strip()):
                return True
        # Check whole string
        return self._looks_like_b64_data(s)


    def _decode_b64(self):
        raw = self.b64_input.get('1.0', 'end').strip()
        if not raw:
            messagebox.showwarning("Empty", "Paste base64 data into the left panel first.")
            return
        try:
            # Strip HL7 ED prefix if present (^AP^PDF^Base64^<data>)
            extracted = self._extract_base64(raw)
            # Strip whitespace for decoding
            clean = re.sub(r'\s+', '', extracted)
            # Pad if needed
            pad = len(clean) % 4
            if pad:
                clean += '=' * (4 - pad)
            decoded = base64.b64decode(clean)
            self._b64_decoded_bytes = decoded
            # Detect type
            is_pdf = decoded[:4] == b'%PDF'
            size_str = f"{len(decoded):,} bytes"
            if is_pdf:
                self.b64_info_var.set(f"✔ PDF detected · {size_str} · click 'Open PDF in Browser' to view")
                self._show_b64_output(f"[PDF Document — {size_str}]\n\nClick '📄 Open PDF in Browser' to view the document.\n\nPDF Header: {decoded[:20]!r}")
            else:
                # Try as text
                try:
                    text = decoded.decode('utf-8')
                    self.b64_info_var.set(f"✔ Text/UTF-8 decoded · {size_str}")
                    self._show_b64_output(text)
                except UnicodeDecodeError:
                    self.b64_info_var.set(f"✔ Binary data decoded · {size_str}")
                    hex_preview = decoded[:512].hex()
                    self._show_b64_output(f"[Binary data — {size_str}]\n\nHex preview (first 512 bytes):\n"
                                         + '\n'.join(hex_preview[i:i+64] for i in range(0, len(hex_preview), 64)))
        except Exception as e:
            messagebox.showerror("Decode Error", f"Could not decode base64:\n{e}")

    def _show_b64_output(self, text):
        self.b64_output.config(state='normal')
        self.b64_output.delete('1.0', 'end')
        self.b64_output.insert('1.0', text)
        self.b64_output.config(state='disabled')

    def _open_pdf_browser(self):
        if not self._b64_decoded_bytes:
            # Try to decode first
            self._decode_b64()
        if not self._b64_decoded_bytes:
            return
        if self._b64_decoded_bytes[:4] != b'%PDF':
            messagebox.showwarning("Not a PDF", "Decoded data does not appear to be a PDF.")
            return
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp.write(self._b64_decoded_bytes)
            tmp.close()
            self._pdf_temp_files.append(tmp.name)
            webbrowser.open(f"file://{tmp.name}")
            self.status_var.set(f"✔ PDF opened in browser: {tmp.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")

    def _save_decoded(self):
        if not self._b64_decoded_bytes:
            messagebox.showwarning("Nothing", "Decode data first.")
            return
        is_pdf = self._b64_decoded_bytes[:4] == b'%PDF'
        ft = [("PDF Files", "*.pdf"), ("All Files", "*.*")] if is_pdf else [("All Files", "*.*")]
        path = filedialog.asksaveasfilename(
            title="Save Decoded File",
            defaultextension=".pdf" if is_pdf else ".bin",
            filetypes=ft)
        if path:
            with open(path, 'wb') as f:
                f.write(self._b64_decoded_bytes)
            self.status_var.set(f"✔ Saved: {path}")

    def _clear_b64(self):
        self.b64_input.delete('1.0', 'end')
        self._show_b64_output("")
        self.b64_info_var.set("Decoded output will appear here")
        self._b64_decoded_bytes = None

    def _import_obx5_b64(self):
        """Pull base64/ED data from OBX segments of current parsed message.
        Scans all fields of each OBX segment and handles HL7 ED format."""
        if not self.parsed_segments:
            messagebox.showwarning("No Message", "Parse an HL7 message first.")
            return
        values = []
        for seg in self.parsed_segments:
            if seg['name'] != 'OBX':
                continue
            # Scan all fields for base64 or ED-encoded data
            for f_idx in range(1, len(seg['fields'])):
                raw = self._field_to_raw(seg['fields'][f_idx])
                if raw and self._looks_like_base64(raw):
                    values.append(raw)
                    break  # only first matching field per OBX
        if not values:
            messagebox.showinfo(
                "Not Found",
                "No base64 or HL7 ED (Encapsulated Data) found in any OBX segment.\n\n"
                "Tip: Base64 in OBX is usually in OBX-5 with ED type\n"
                "(e.g. ^AP^PDF^Base64^<data>).")
            return
        self.b64_input.delete('1.0', 'end')
        self.b64_input.insert('1.0', values[0])
        self.main_nb.select(2)  # Switch to Base64 tab
        self.status_var.set(f"✔ Imported base64/ED data from OBX ({len(values)} OBX segment(s) found)")
        # Auto-decode immediately
        self._decode_b64()

    def _view_selected_field_as_pdf(self):
        """View currently selected field value as PDF (handles HL7 ED format)."""
        raw = self.field_editor.get().strip()
        if not raw:
            return
        self.b64_input.delete('1.0', 'end')
        self.b64_input.insert('1.0', raw)
        self.main_nb.select(2)
        self._decode_b64()

    # =========================================================================
    # NEW: Compare logic
    # =========================================================================
    def _load_compare_from_editor(self):
        content = self.output_text.get('1.0', 'end').strip() or self.input_text.get('1.0', 'end').strip()
        if content:
            self.cmp_a_text.delete('1.0', 'end')
            self.cmp_a_text.insert('1.0', content)
            self.main_nb.select(1)
            self.status_var.set("✔ Loaded current message into Compare A")

    def _do_compare(self):
        a = self.cmp_a_text.get('1.0', 'end').strip()
        b = self.cmp_b_text.get('1.0', 'end').strip()
        if not a or not b:
            messagebox.showwarning("Compare", "Paste both messages (A and B) to compare.")
            return

        # Normalize line endings
        def lines(msg):
            return msg.replace('\r\n', '\n').replace('\r', '\n').splitlines()

        a_lines = lines(a)
        b_lines = lines(b)

        self.cmp_result.config(state='normal')
        self.cmp_result.delete('1.0', 'end')

        added = deleted = changed = same = 0
        matcher = difflib.SequenceMatcher(None, a_lines, b_lines)

        for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
            if opcode == 'equal':
                for line in a_lines[a1:a2]:
                    self.cmp_result.insert('end', f"   {line}\n", 'same')
                    same += 1
            elif opcode == 'delete':
                for line in a_lines[a1:a2]:
                    self.cmp_result.insert('end', f"−  {line}\n", 'del')
                    deleted += 1
            elif opcode == 'insert':
                for line in b_lines[b1:b2]:
                    self.cmp_result.insert('end', f"+  {line}\n", 'add')
                    added += 1
            elif opcode == 'replace':
                for line in a_lines[a1:a2]:
                    self.cmp_result.insert('end', f"≠  {line}\n", 'chg')
                    changed += 1
                for line in b_lines[b1:b2]:
                    self.cmp_result.insert('end', f"→  {line}\n", 'add')

        self.cmp_result.config(state='disabled')
        total_diff = added + deleted + changed
        self.cmp_summary_var.set(
            f"  {same} identical  ·  {deleted} removed  ·  {added} added  ·  {changed} changed  "
            f"·  {'✔ IDENTICAL' if total_diff == 0 else f'{total_diff} differences'}")
        self.status_var.set(f"✔ Compare complete · {total_diff} difference(s) found")

    def _clear_compare(self):
        self.cmp_a_text.delete('1.0', 'end')
        self.cmp_b_text.delete('1.0', 'end')
        self.cmp_result.config(state='normal')
        self.cmp_result.delete('1.0', 'end')
        self.cmp_result.config(state='disabled')
        self.cmp_summary_var.set("")

    # =========================================================================
    # NEW: Statistics logic
    # =========================================================================
    def _refresh_stats(self):
        if not self.parsed_segments:
            messagebox.showinfo("No Data", "Parse an HL7 message first.")
            return

        # Clear existing cards
        for w in self.stats_cards_frame.winfo_children():
            w.destroy()

        # Compute stats
        total_segs = len(self.parsed_segments)
        total_fields = sum(
            sum(1 for f_idx, _ in enumerate(seg['fields']) if f_idx > 0 and self._field_to_raw(_))
            for seg in self.parsed_segments)
        seg_counts = {}
        for seg in self.parsed_segments:
            seg_counts[seg['name']] = seg_counts.get(seg['name'], 0) + 1

        msh = next((s for s in self.parsed_segments if s['name'] == 'MSH'), None)
        msg_type = ""
        version = ""
        send_app = ""
        recv_app = ""
        if msh:
            try:
                msg_type = self._field_to_raw(msh['fields'][10]) if len(msh['fields']) > 10 else ''
                version  = self._field_to_raw(msh['fields'][13]) if len(msh['fields']) > 13 else ''
                send_app = self._field_to_raw(msh['fields'][3])  if len(msh['fields']) > 3 else ''
                recv_app = self._field_to_raw(msh['fields'][5])  if len(msh['fields']) > 5 else ''
            except Exception:
                pass

        # Summary cards
        cards = [
            ("Segments", str(total_segs), self.colors['accent']),
            ("Non-empty Fields", str(total_fields), self.colors['accent2']),
            ("Unique Seg Types", str(len(seg_counts)), self.colors['accent3']),
            ("Message Type", msg_type or "—", self.colors['warning']),
            ("HL7 Version", version or "—", self.colors['text']),
            ("Sending App", send_app or "—", self.colors['text_muted']),
            ("Receiving App", recv_app or "—", self.colors['text_muted']),
        ]
        for title, value, color in cards:
            card = tk.Frame(self.stats_cards_frame, bg=self.colors['panel'],
                            relief='flat', bd=1)
            card.pack(side='left', padx=6, pady=4, ipadx=12, ipady=8)
            tk.Label(card, text=value, bg=self.colors['panel'], fg=color,
                     font=('Segoe UI', 16, 'bold')).pack()
            tk.Label(card, text=title, bg=self.colors['panel'], fg=self.colors['text_muted'],
                     font=('Segoe UI', 8)).pack()

        # Frequency table
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        for name, count in sorted(seg_counts.items(), key=lambda x: -x[1]):
            desc = HL7_SEGMENT_DESCRIPTIONS.get(name, 'Custom/Z-Segment')
            notes = ""
            if name == 'OBX':
                types = []
                for seg in self.parsed_segments:
                    if seg['name'] == 'OBX' and len(seg['fields']) > 2:
                        vt = self._field_to_raw(seg['fields'][2])
                        if vt and vt not in types:
                            types.append(vt)
                notes = "Value types: " + ", ".join(types) if types else ""
            self.stats_tree.insert('', 'end', values=(count, desc, notes), text=name)

        self.status_var.set(f"✔ Statistics refreshed · {total_segs} segments · {total_fields} fields")


def main():
    root = tk.Tk()
    app = HL7ViewerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
