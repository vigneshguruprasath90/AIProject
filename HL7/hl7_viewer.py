#!/usr/bin/env python3
"""
HL7 Viewer & Editor
-------------------
A free, open-source HL7 message viewer and editor for personal/internal use.
No license restrictions. Share freely with your team.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import re
import json
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

SAMPLE_HL7 = """MSH|^~\\&|HIS|HOSPITAL|LAB|LABORATORY|20231015120000||ORU^R01|MSG001|P|2.5|||AL|NE
PID|1||123456^^^MRN||DOE^JOHN^A||19800101|M|||123 MAIN ST^^SPRINGFIELD^IL^62701^USA||555-555-1234||EN|S|||SSN123456789
PV1|1|I|ICU^101^A||||1234^SMITH^JOHN^A|||SUR|||||||1234^SMITH^JOHN^A|IP||||||||||||||||||||||ADM|20231015080000
OBR|1|ORD001|FILL001|85025^CBC^L|||20231015100000|||||||20231015110000||1234^SMITH^JOHN^A|||||20231015115000||HM|F
OBX|1|NM|6690-2^WBC^LN||7.5|10*3/uL|4.5-11.0|N|||F|||20231015115000
OBX|2|NM|789-8^RBC^LN||4.8|10*6/uL|4.5-5.5|N|||F|||20231015115000
OBX|3|NM|718-7^HGB^LN||14.5|g/dL|13.5-17.5|N|||F|||20231015115000
OBX|4|NM|4544-3^HCT^LN||43.2|%|41.0-53.0|N|||F|||20231015115000
NTE|1||Patient fasting for 8 hours prior to collection"""


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
            fields = line.split(self.field_sep)
            parsed_fields = []
            for i, field in enumerate(fields):
                reps = field.split(self.repetition_sep)
                parsed_reps = []
                for rep in reps:
                    comps = rep.split(self.component_sep)
                    parsed_comps = []
                    for comp in comps:
                        subs = comp.split(self.subcomponent_sep)
                        parsed_comps.append(subs)
                    parsed_reps.append(parsed_comps)
                parsed_fields.append(parsed_reps)
            segments.append({'name': seg_name, 'raw': line, 'fields': parsed_fields})
        return segments

    def get_field_desc(self, seg_name, field_idx):
        descs = SEGMENT_FIELD_DESCRIPTIONS.get(seg_name, {})
        return descs.get(field_idx, f"Field {field_idx}")

    def rebuild(self, segments):
        lines = []
        for seg in segments:
            lines.append(seg['raw'])
        return '\r\n'.join(lines)


class HL7ViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HL7 Viewer & Editor")
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)
        self.parser = HL7Parser()
        self.parsed_segments = []
        self.selected_segment_idx = None
        self.selected_field_idx = None
        self._setup_theme()
        self._build_ui()
        self._load_sample()

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
            'msh': '#2D1F5E',
            'pid': '#1A3A2A',
            'obx': '#3A2A10',
            'obr': '#2A1A3A',
            'pv1': '#1A2A3A',
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

    def _build_ui(self):
        self.root.configure(bg=self.colors['bg'])
        # Top bar
        topbar = tk.Frame(self.root, bg=self.colors['panel'], height=50)
        topbar.pack(fill='x', side='top')
        topbar.pack_propagate(False)
        logo = tk.Label(topbar, text="⚕ HL7 Viewer & Editor",
                        bg=self.colors['panel'], fg=self.colors['accent'],
                        font=('Segoe UI', 13, 'bold'), padx=16)
        logo.pack(side='left', pady=8)
        version_lbl = tk.Label(topbar, text="v1.0 · Free & Open Source",
                               bg=self.colors['panel'], fg=self.colors['text_muted'],
                               font=('Segoe UI', 9))
        version_lbl.pack(side='left', pady=8)
        # Right toolbar buttons
        btn_frame = tk.Frame(topbar, bg=self.colors['panel'])
        btn_frame.pack(side='right', padx=12, pady=8)
        ttk.Button(btn_frame, text="📂 Open File", command=self._open_file, style='TButton').pack(side='left', padx=3)
        ttk.Button(btn_frame, text="💾 Save", command=self._save_file, style='TButton').pack(side='left', padx=3)
        ttk.Button(btn_frame, text="📋 Copy Output", command=self._copy_output, style='TButton').pack(side='left', padx=3)
        ttk.Button(btn_frame, text="🗑 Clear", command=self._clear, style='TButton').pack(side='left', padx=3)
        ttk.Button(btn_frame, text="💡 Sample", command=self._load_sample, style='TButton').pack(side='left', padx=3)
        # Status bar
        self.status_var = tk.StringVar(value="Ready · Paste HL7 message to begin")
        statusbar = tk.Frame(self.root, bg=self.colors['panel'], height=26)
        statusbar.pack(fill='x', side='bottom')
        statusbar.pack_propagate(False)
        self.status_lbl = tk.Label(statusbar, textvariable=self.status_var,
                                   bg=self.colors['panel'], fg=self.colors['text_muted'],
                                   font=('Segoe UI', 9), padx=12)
        self.status_lbl.pack(side='left', pady=3)
        self.msg_type_lbl = tk.Label(statusbar, text="",
                                     bg=self.colors['panel'], fg=self.colors['accent2'],
                                     font=('Segoe UI', 9, 'bold'), padx=12)
        self.msg_type_lbl.pack(side='right', pady=3)
        # Main paned window
        main_pane = ttk.PanedWindow(self.root, orient='horizontal')
        main_pane.pack(fill='both', expand=True, padx=0, pady=0)
        # Left: input/output notebook
        left_frame = ttk.Frame(main_pane, style='Panel.TFrame')
        main_pane.add(left_frame, weight=45)
        nb = ttk.Notebook(left_frame)
        nb.pack(fill='both', expand=True, padx=0, pady=0)
        # Input tab
        input_frame = ttk.Frame(nb)
        nb.add(input_frame, text="  📥 Input  ")
        input_lbl = tk.Label(input_frame, text="Paste HL7 Message Here:",
                             bg=self.colors['panel'], fg=self.colors['text_muted'],
                             font=('Segoe UI', 9), pady=4)
        input_lbl.pack(anchor='w', padx=8)
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
        parse_btn = ttk.Button(input_frame, text="▶ Parse Message  (Ctrl+Enter)",
                               command=self._parse_message, style='Accent.TButton')
        parse_btn.pack(fill='x', padx=8, pady=6)
        # Output tab
        output_frame = ttk.Frame(nb)
        nb.add(output_frame, text="  📤 Output  ")
        out_lbl = tk.Label(output_frame, text="Rebuilt / Edited HL7:",
                           bg=self.colors['panel'], fg=self.colors['text_muted'],
                           font=('Segoe UI', 9), pady=4)
        out_lbl.pack(anchor='w', padx=8)
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
        # Right: structured view + field editor pane
        right_pane = ttk.PanedWindow(main_pane, orient='vertical')
        main_pane.add(right_pane, weight=55)
        # Top right: segment tree view
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
        # Bottom right: field editor
        edit_frame = ttk.Frame(right_pane, style='Panel.TFrame')
        right_pane.add(edit_frame, weight=50)
        edit_hdr = tk.Frame(edit_frame, bg=self.colors['seg_header'], height=32)
        edit_hdr.pack(fill='x')
        edit_hdr.pack_propagate(False)
        tk.Label(edit_hdr, text="  ✏ Field Editor & Component View",
                 bg=self.colors['seg_header'], fg=self.colors['text'],
                 font=('Segoe UI', 10, 'bold')).pack(side='left', pady=6)
        edit_content = tk.Frame(edit_frame, bg=self.colors['panel'])
        edit_content.pack(fill='both', expand=True, padx=8, pady=6)
        # Field info row
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
        # Raw field value editor
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
        # Component breakdown
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
        # Repetitions label
        self.rep_var = tk.StringVar(value="")
        tk.Label(edit_content, textvariable=self.rep_var,
                 bg=self.colors['panel'], fg=self.colors['warning'],
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(4, 0))

    def _on_input_change(self, event=None):
        content = self.input_text.get('1.0', 'end').strip()
        if content:
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
            self.status_var.set(f"✔ Parsed {len(self.parsed_segments)} segments successfully · {datetime.now().strftime('%H:%M:%S')}")
            self.nb.select(0)
        except Exception as e:
            self.status_var.set(f"✖ Parse error: {str(e)}")
            messagebox.showerror("Parse Error", str(e))

    def _detect_msg_type(self):
        for seg in self.parsed_segments:
            if seg['name'] == 'MSH':
                fields = seg['fields']
                if len(fields) > 9:
                    try:
                        msg_type = fields[9][0][0][0] if fields[9] else ''
                        trigger = fields[9][0][1][0] if (fields[9] and len(fields[9][0]) > 1) else ''
                        version = fields[12][0][0][0] if (len(fields) > 12 and fields[12]) else ''
                        if msg_type:
                            self.msg_type_lbl.config(
                                text=f"  {msg_type}^{trigger}  |  HL7 v{version}  ")
                    except:
                        pass

    def _populate_tree(self):
        for item in self.seg_tree.get_children():
            self.seg_tree.delete(item)
        self.seg_count_lbl.config(text=f"{len(self.parsed_segments)} segments")
        for s_idx, seg in enumerate(self.parsed_segments):
            seg_name = seg['name']
            seg_desc = HL7_SEGMENT_DESCRIPTIONS.get(seg_name, "Custom/Unknown Segment")
            seg_node = self.seg_tree.insert('', 'end',
                                            text=f"  {seg_name}",
                                            values=('', '', seg_desc),
                                            tags=('segment',),
                                            iid=f"seg_{s_idx}")
            fields = seg['fields']
            start = 2 if seg_name == 'MSH' else 1
            for f_idx, field in enumerate(fields):
                actual_idx = f_idx if seg_name != 'MSH' else f_idx
                if seg_name == 'MSH' and f_idx == 0:
                    continue
                if seg_name == 'MSH' and f_idx == 1:
                    continue
                disp_idx = f_idx if seg_name != 'MSH' else f_idx
                raw_val = self._field_to_raw(field)
                desc = self.parser.get_field_desc(seg_name, f_idx)
                tag = 'field' if f_idx % 2 == 0 else 'field_alt'
                if not raw_val:
                    tag = 'empty'
                self.seg_tree.insert(seg_node, 'end',
                                     text=f"   {seg_name}",
                                     values=(f"[{f_idx}]", raw_val[:80] + ('…' if len(raw_val) > 80 else ''), desc),
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

    def _on_tree_select(self, event):
        sel = self.seg_tree.selection()
        if not sel:
            return
        iid = sel[0]
        parts = iid.split('_')
        if len(parts) == 2:
            # Segment node selected — show segment raw
            s_idx = int(parts[1])
            seg = self.parsed_segments[s_idx]
            self.field_path_var.set(f"Segment: {seg['name']}  (row {s_idx + 1})")
            self.field_desc_var.set(HL7_SEGMENT_DESCRIPTIONS.get(seg['name'], ''))
            self.field_editor.delete(0, 'end')
            self.field_editor.insert(0, seg['raw'])
            self.selected_segment_idx = s_idx
            self.selected_field_idx = None
            self._clear_comp_tree()
        elif len(parts) == 4:
            # Field node
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
            # Show repetitions
            if len(field) > 1:
                self.rep_var.set(f"⟳ {len(field)} repetition(s) — separated by ~")
            else:
                self.rep_var.set("")

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
                    parsed_comps = []
                    for comp in comps:
                        subs = comp.split(self.parser.subcomponent_sep)
                        parsed_comps.append(subs)
                    parsed_reps.append(parsed_comps)
                parsed_fields.append(parsed_reps)
            seg['fields'] = parsed_fields
        else:
            f_idx = self.selected_field_idx
            reps = new_val.split(self.parser.repetition_sep)
            parsed_reps = []
            for rep in reps:
                comps = rep.split(self.parser.component_sep)
                parsed_comps = []
                for comp in comps:
                    subs = comp.split(self.parser.subcomponent_sep)
                    parsed_comps.append(subs)
                parsed_reps.append(parsed_comps)
            seg['fields'][f_idx] = parsed_reps
            # Rebuild raw
            raw_fields = []
            for fld in seg['fields']:
                raw_fields.append(self._field_to_raw(fld))
            seg['raw'] = self.parser.field_sep.join(raw_fields)
        self._populate_tree()
        self._update_output()
        self.status_var.set(f"✔ Field updated successfully · {datetime.now().strftime('%H:%M:%S')}")
        # Re-select same field
        try:
            if self.selected_field_idx is not None:
                self.seg_tree.selection_set(f"seg_{s_idx}_fld_{self.selected_field_idx}")
            else:
                self.seg_tree.selection_set(f"seg_{s_idx}")
        except:
            pass

    def _reset_field(self):
        sel = self.seg_tree.selection()
        if not sel:
            return
        iid = sel[0]
        parts = iid.split('_')
        if len(parts) == 4:
            s_idx = int(parts[1])
            f_idx = int(parts[3])
            seg = self.parsed_segments[s_idx]
            raw = self._field_to_raw(seg['fields'][f_idx])
            self.field_editor.delete(0, 'end')
            self.field_editor.insert(0, raw)

    def _update_output(self):
        lines = []
        for seg in self.parsed_segments:
            lines.append(seg['raw'])
        result = '\r\n'.join(lines)
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
            filetypes=[("HL7 Files", "*.hl7 *.txt *.msg"), ("All Files", "*.*")]
        )
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
        content = self.output_text.get('1.0', 'end').strip()
        if not content:
            content = self.input_text.get('1.0', 'end').strip()
        if not content:
            messagebox.showwarning("Nothing to Save", "No HL7 content to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save HL7 File",
            defaultextension=".hl7",
            filetypes=[("HL7 Files", "*.hl7"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
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
        self.status_var.set("Cleared · Paste HL7 message to begin")

    def _load_sample(self):
        self._clear()
        self.input_text.insert('1.0', SAMPLE_HL7)
        self._parse_message(SAMPLE_HL7)
        self.status_var.set("✔ Sample HL7 ORU^R01 message loaded")


def main():
    root = tk.Tk()
    app = HL7ViewerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
