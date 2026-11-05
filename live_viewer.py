#!/usr/bin/env python3
"""Interactive Revo3 URDF mirror driven only by measured RS485 feedback."""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders
from OpenGL.GL import GL_DEPTH_TEST
import trimesh

from live_feedback import FeedbackReader
from live_model import HandModel, JOINT_LABELS, DISPLAY_ORDER, ROOT

FINGER_COLORS = dict(thumb=(1., .57, .25, 1.), index=(.27, .7, 1., 1.),
                     middle=(.35, .83, .68, 1.), ring=(.7, .58, 1., 1.), little=(.96, .49, .65, 1.))

# Camera-relative fill lighting keeps the joint surfaces readable from either side.
shaders.ShaderProgram('revo_lit', [shaders.VertexShader('''
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec4 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_color = a_color;
        gl_Position = u_mvp * a_position;
    }
'''), shaders.FragmentShader('''
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        vec3 n = normalize(v_normal);
        float key = abs(dot(n, normalize(vec3(0.4, -0.6, 1.0))));
        float fill = abs(dot(n, normalize(vec3(-0.8, 0.3, 0.5))));
        gl_FragColor = vec4(v_color.rgb * (0.40 + 0.45 * key + 0.15 * fill), v_color.a);
    }
''')])


class Viewer(W.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reader = None
        self.metadata = {}
        self.displayed = None
        self.reference = None
        self.last_arrival = None
        self.last_error = ''
        self.control = {'mode': 'disconnected', 'hold_targets_deg': None}
        self.mode_pending = False
        self.received_count = 0
        self.signs = np.ones(21)
        self.offsets = np.zeros(21)
        self.selected = 19
        self.mesh_items = []
        self.model = None
        self.poses = None
        self.axis_data = None
        self.setWindowTitle('Revo3 · Live URDF Mirror')
        self.resize(1440, 920)
        self.build_ui()
        self.load_model(getattr(args, 'force_side', None) or args.side)
        self.refresh_ports()
        if args.port:
            self.port.setCurrentText(args.port)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        if args.demo or args.connect:
            QtCore.QTimer.singleShot(300, lambda: self.connect_reader(args.demo))
        if args.smoke_test:
            QtCore.QTimer.singleShot(4000, self.smoke_test)

    def build_ui(self):
        root = W.QWidget()
        self.setCentralWidget(root)
        layout = W.QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 14)
        header = W.QHBoxLayout()
        heading = W.QLabel('REVO3  /  LIVE MIRROR')
        heading.setObjectName('title')
        header.addWidget(heading)
        header.addStretch()
        self.badge = W.QLabel('DISCONNECTED')
        self.badge.setObjectName('badge')
        header.addWidget(self.badge)
        layout.addLayout(header)
        self.identity = W.QLabel('Live joint feedback · connects in zero-force · normal force holds the newly measured pose.')
        self.identity.setObjectName('muted')
        layout.addWidget(self.identity)
        connection = W.QHBoxLayout()
        connection.addWidget(W.QLabel('RS485'))
        self.port = W.QComboBox()
        self.port.setEditable(True)
        self.port.setMinimumWidth(220)
        connection.addWidget(self.port, 1)
        refresh = W.QPushButton('Refresh ports')
        refresh.clicked.connect(self.refresh_ports)
        connection.addWidget(refresh)
        connection.addWidget(W.QLabel('Baud'))
        self.baud = W.QComboBox()
        self.baud.addItems(['5000000', '3000000', '2000000', '1000000'])
        self.baud.setCurrentText(str(self.args.baudrate))
        connection.addWidget(self.baud)
        connection.addWidget(W.QLabel('Device ID'))
        self.device_id = W.QSpinBox()
        self.device_id.setRange(1, 247)
        self.device_id.setValue(self.args.slave_id)
        connection.addWidget(self.device_id)
        self.connect_button = W.QPushButton('Connect')
        self.connect_button.setObjectName('primary')
        self.connect_button.clicked.connect(lambda: self.connect_reader(False))
        connection.addWidget(self.connect_button)
        self.disconnect_button = W.QPushButton('Disconnect')
        self.disconnect_button.clicked.connect(self.disconnect_reader)
        self.disconnect_button.setEnabled(False)
        connection.addWidget(self.disconnect_button)
        self.demo_button = W.QPushButton('Demo')
        self.demo_button.clicked.connect(lambda: self.connect_reader(True))
        connection.addWidget(self.demo_button)
        layout.addLayout(connection)

        force_bar = W.QHBoxLayout()
        self.zero_force = W.QPushButton('Zero-force ON')
        self.zero_force.setCheckable(True)
        self.zero_force.setChecked(True)
        self.zero_force.setEnabled(False)
        self.zero_force.setMinimumWidth(170)
        self.zero_force.toggled.connect(self.request_force_mode)
        force_bar.addWidget(self.zero_force)
        self.force_label = W.QLabel('Default: zero-force on connection. Switch OFF to hold the current physical pose.')
        self.force_label.setWordWrap(True)
        force_bar.addWidget(self.force_label, 1)
        layout.addLayout(force_bar)

        split = W.QSplitter()
        layout.addWidget(split, 1)
        scene_panel = W.QWidget()
        scene_layout = W.QVBoxLayout(scene_panel)
        scene_layout.setContentsMargins(0, 8, 8, 0)
        toolbar = W.QHBoxLayout()
        for text, azimuth, elevation in [('Palm', 0, 5), ('Back', 180, 5), ('Side', 90, 5), ('Top', 0, 85)]:
            button = W.QPushButton(text)
            button.clicked.connect(lambda checked=False, a=azimuth, e=elevation: self.camera(a, e))
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.focus = W.QComboBox()
        self.focus.addItems(['All fingers', 'thumb', 'index', 'middle', 'ring', 'little', 'thumb + index'])
        self.focus.currentTextChanged.connect(self.recolor)
        toolbar.addWidget(self.focus)
        scene_layout.addLayout(toolbar)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#101a29')
        self.view.opts['center'] = QtGui.QVector3D(0, 0, .115)
        self.camera(15, 15)
        scene_layout.addWidget(self.view, 1)
        grid = gl.GLGridItem()
        grid.setSize(.3, .3)
        grid.setSpacing(.025, .025)
        grid.setColor((70, 90, 115, 100))
        self.view.addItem(grid)
        for vector, color in [([.035, 0, 0], (1., .3, .3, 1.)), ([0, .035, 0], (.3, 1., .4, 1.)),
                              ([0, 0, .035], (.3, .6, 1., 1.))]:
            self.view.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], vector]), color=color, width=2))
        self.axis_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1., .87, .35, 1.), width=4)
        self.axis_dot = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1., .87, .35, 1.), size=10)
        self.view.addItem(self.axis_line)
        self.view.addItem(self.axis_dot)
        for item in (self.axis_line, self.axis_dot):
            item.updateGLOptions({GL_DEPTH_TEST: False})
            item.setDepthValue(100)
        self.ref_lines = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(.4, .92, 1., .75), width=2, mode='lines')
        self.view.addItem(self.ref_lines)
        self.ref_lines.hide()
        help_text = W.QLabel('Drag to orbit · wheel to zoom · Ctrl + drag to pan · yellow line = selected joint axis')
        help_text.setObjectName('muted')
        scene_layout.addWidget(help_text)
        actions = W.QHBoxLayout()
        self.freeze = W.QPushButton('Freeze view')
        self.freeze.setCheckable(True)
        self.freeze.toggled.connect(lambda frozen: self.freeze.setText('Resume live view' if frozen else 'Freeze view'))
        actions.addWidget(self.freeze)
        self.pin_button = W.QPushButton('Pin reference')
        self.pin_button.clicked.connect(self.pin_reference)
        actions.addWidget(self.pin_button)
        self.capture_button = W.QPushButton('Save snapshot')
        self.capture_button.setObjectName('primary')
        self.capture_button.clicked.connect(self.capture)
        actions.addWidget(self.capture_button)
        scene_layout.addLayout(actions)
        split.addWidget(scene_panel)

        panel = W.QWidget()
        side_layout = W.QVBoxLayout(panel)
        side_layout.setContentsMargins(8, 8, 0, 0)
        self.model_label = W.QLabel()
        self.model_label.setObjectName('section')
        side_layout.addWidget(self.model_label)
        self.table = W.QTableWidget(21, 6)
        self.table.setHorizontalHeaderLabels(['Motor', 'URDF joint', 'Angle °', 'Δ ref °', 'mA', 'Flags'])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(W.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setMinimumSectionSize(21)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, W.QHeaderView.ResizeMode.Stretch)
        for row, motor in enumerate(DISPLAY_ORDER):
            for col, text in enumerate([f'M{motor}', JOINT_LABELS[motor], '—', '—', '—', '—']):
                self.table.setItem(row, col, W.QTableWidgetItem(text))
        self.table.itemSelectionChanged.connect(self.selection_changed)
        side_layout.addWidget(self.table, 1)
        self.selected_label = W.QLabel()
        self.selected_label.setWordWrap(True)
        side_layout.addWidget(self.selected_label)
        axis_controls = W.QHBoxLayout()
        self.show_axis = W.QCheckBox('Show joint axis')
        self.show_axis.setChecked(True)
        self.show_axis.toggled.connect(self.update_axis)
        axis_controls.addWidget(self.show_axis)
        self.vendor_colors = W.QCheckBox('Original materials')
        self.vendor_colors.toggled.connect(self.recolor)
        axis_controls.addWidget(self.vendor_colors)
        side_layout.addLayout(axis_controls)

        preview_box = W.QGroupBox('Alignment preview · display only')
        preview_layout = W.QVBoxLayout(preview_box)
        self.preview = W.QCheckBox('Apply sign / offset preview')
        self.preview.toggled.connect(self.redraw)
        preview_layout.addWidget(self.preview)
        correction = W.QHBoxLayout()
        correction.addWidget(W.QLabel('Selected joint sign'))
        self.sign = W.QComboBox()
        self.sign.addItems(['+1', '−1'])
        correction.addWidget(self.sign)
        correction.addWidget(W.QLabel('Offset °'))
        self.offset = W.QDoubleSpinBox()
        self.offset.setRange(-180, 180)
        self.offset.setDecimals(2)
        self.offset.setSingleStep(.5)
        correction.addWidget(self.offset)
        self.sign.currentIndexChanged.connect(self.correction_changed)
        self.offset.valueChanged.connect(self.correction_changed)
        preview_layout.addLayout(correction)
        reset = W.QPushButton('Reset all preview corrections')
        reset.clicked.connect(self.reset_corrections)
        preview_layout.addWidget(reset)
        side_layout.addWidget(preview_box)
        note = W.QLabel('Angle column = raw feedback in degrees. No smoothing or clamping.\n'
                        'LIM = outside URDF limits · E = device error bits.\n'
                        'Zero-force ON: move by hand. OFF: hold the newly measured pose.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        side_layout.addWidget(note)
        split.addWidget(panel)
        split.setSizes([850, 550])
        self.status = W.QLabel('Ready. Connection starts in zero-force for all 21 joints.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table.selectRow(0)
        self.setStyleSheet('''
            QMainWindow, QWidget { background: #101722; color: #d9e4f2; font-size: 12px; }
            QLabel#title { font-size: 23px; font-weight: 700; letter-spacing: 2px; }
            QLabel#section { font-size: 15px; font-weight: 600; padding: 3px; }
            QLabel#muted { color: #8fabc4; font-size: 11px; }
            QLabel#badge { border-radius: 5px; background: #25364c; padding: 6px 14px; font-weight: 700; }
            QPushButton { background: #223249; border: 1px solid #344a65; border-radius: 5px; padding: 7px 12px; }
            QPushButton:hover { background: #304861; }
            QPushButton:checked { background: #66512a; }
            QPushButton:disabled { color: #5a6b7b; background: #18222f; }
            QPushButton#primary { background: #246b7a; border-color: #368b9c; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #1c2b3d; padding: 5px; border: 1px solid #344a65; border-radius: 4px; }
            QTableWidget { background: #152131; alternate-background-color: #192739; gridline-color: #25374b; border: 1px solid #2b3d53; }
            QTableWidget::item:selected { background: #304d69; }
            QHeaderView::section { background: #223249; padding: 6px; border: 0; color: #9bb6d0; }
            QGroupBox { border: 1px solid #30435b; border-radius: 5px; margin-top: 9px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9bb6d0; }
            QSplitter::handle { background: #243448; width: 2px; }
        ''')

    def camera(self, azimuth, elevation):
        self.view.setCameraPosition(distance=.33, elevation=elevation, azimuth=azimuth)

    def refresh_ports(self):
        previous = self.port.currentText()
        ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
        self.port.clear()
        self.port.addItems(ports)
        if previous:
            self.port.setCurrentText(previous)

    def load_model(self, side):
        if self.model is not None and self.model.side == side:
            return
        model = HandModel(side)
        new_items = []
        for visual in model.visuals:
            mesh = trimesh.load_mesh(visual['path'], process=False)
            vertices = np.asarray(mesh.vertices * visual['scale'], dtype=np.float32)
            item = gl.GLMeshItem(vertexes=vertices, faces=np.asarray(mesh.faces, dtype=np.uint32),
                                 smooth=False, shader='revo_lit', drawEdges=False, computeNormals=True)
            new_items.append((visual, item))
        for _, item in self.mesh_items:
            self.view.removeItem(item)
        self.mesh_items = new_items
        self.model = model
        self.setWindowTitle(f'Revo3 {side.upper()} · Live URDF Mirror')
        for _, item in new_items:
            self.view.addItem(item)
        self.reference = None
        self.ref_lines.hide()
        self.model_label.setText(f'{side.upper()} HAND  ·  21 joints  ·  official 20261014 model')
        self.recolor()
        self.apply_pose(np.zeros(21))

    def recolor(self, *unused):
        focus = self.focus.currentText()
        for visual, item in self.mesh_items:
            finger = next((f for f in FINGER_COLORS if f in visual['link']), None)
            visible = focus == 'All fingers' or finger is None or finger in focus
            item.setVisible(visible)
            color = visual['color'] if self.vendor_colors.isChecked() or finger is None else FINGER_COLORS[finger]
            color = np.array(color, dtype=float)
            if not self.vendor_colors.isChecked() and ('rubber' in visual['link'] or 'touch' in visual['link']):
                color[:3] *= .65
            item.setColor(tuple(color))

    def apply_pose(self, degrees):
        self.poses, self.axis_data = self.model.forward(degrees)
        for visual, item in self.mesh_items:
            matrix = self.poses[visual['link']] @ visual['origin']
            item.setTransform(QtGui.QMatrix4x4(*matrix.flatten().tolist()))
        self.update_axis()

    def selection_changed(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self.selected = DISPLAY_ORDER[row]
        if hasattr(self, 'sign'):
            self.sign.blockSignals(True)
            self.offset.blockSignals(True)
            self.sign.setCurrentIndex(0 if self.signs[self.selected] > 0 else 1)
            self.offset.setValue(self.offsets[self.selected])
            self.sign.blockSignals(False)
            self.offset.blockSignals(False)
        self.update_axis()

    def update_axis(self, *unused):
        if self.axis_data is None:
            return
        point, axis = self.axis_data[self.selected]
        self.axis_line.setData(pos=np.array([point - axis * .012, point + axis * .028]))
        self.axis_dot.setData(pos=point.reshape(1, 3))
        self.axis_line.setVisible(self.show_axis.isChecked())
        self.axis_dot.setVisible(self.show_axis.isChecked())
        limits = self.model.limits_deg(self.selected)
        raw = self.displayed['positions'][self.selected] if self.displayed else None
        text = f'M{self.selected} → {self.model.names[self.selected]}\nURDF range: {limits[0]:.2f}° to {limits[1]:.2f}°'
        if raw is not None:
            shown = self.model_degrees()[self.selected]
            text += f'  |  displayed: {shown:.2f}° ({np.deg2rad(shown):.4f} rad)'
        self.selected_label.setText(text)

    def model_degrees(self):
        raw = np.array(self.displayed['positions']) if self.displayed else np.zeros(21)
        return raw * self.signs + self.offsets if self.preview.isChecked() else raw

    def correction_changed(self, *unused):
        self.signs[self.selected] = 1 if self.sign.currentIndex() == 0 else -1
        self.offsets[self.selected] = self.offset.value()
        self.redraw()

    def reset_corrections(self):
        self.signs[:] = 1
        self.offsets[:] = 0
        self.preview.setChecked(False)
        self.selection_changed()
        self.redraw()

    def redraw(self, *unused):
        if self.model is not None:
            self.apply_pose(self.model_degrees())
        if not self.displayed:
            return
        for row, motor in enumerate(DISPLAY_ORDER):
            raw = self.displayed['positions'][motor]
            delta = raw - self.reference['positions'][motor] if self.reference else None
            lower, upper = self.model.limits_deg(motor)
            error = self.displayed['errors'][motor]
            flags = []
            if raw < lower or raw > upper:
                flags.append('LIM')
            if error:
                flags.append(f'E:{int(error):04X}')
            for col, value in [(2, f'{raw:.2f}'), (3, f'{delta:+.2f}' if delta is not None else '—'),
                               (4, f"{self.displayed['currents'][motor]:.0f}"), (5, ' '.join(flags) or '—')]:
                self.table.item(row, col).setText(value)
            self.table.item(row, 5).setForeground(QtGui.QColor('#ffbc6e' if flags else '#8fabc4'))
            self.table.item(row, 2).setForeground(QtGui.QColor('#70e2d1' if delta is not None and abs(delta) > 1 else '#d9e4f2'))
            self.table.item(row, 5).setToolTip(f"Status: 0x{int(self.displayed['statuses'][motor]):04X}; error: 0x{int(error):04X}")

    def pin_reference(self):
        if self.displayed is None:
            return
        self.reference = dict(self.displayed)
        self.reference['displayed_pose_rad'] = {
            name: float(np.deg2rad(q)) for name, q in zip(self.model.names, self.model_degrees())}
        points = []
        for joint in self.model.joints:
            if any(f in joint['name'] for f in FINGER_COLORS) and 'rubber' not in joint['name'] and 'touch' not in joint['name']:
                points.extend([self.poses[joint['parent']][:3, 3], self.poses[joint['child']][:3, 3]])
        self.ref_lines.setData(pos=np.array(points))
        self.ref_lines.show()
        self.pin_button.setText('Replace reference')
        self.redraw()

    def connect_reader(self, demo):
        if self.reader is not None:
            return
        port = self.port.currentText().strip()
        if not demo and not port:
            self.status.setText('Select an RS485 serial port first.')
            return
        self.displayed = None
        self.reference = None
        self.ref_lines.hide()
        self.pin_button.setText('Pin reference')
        self.freeze.setChecked(False)
        self.last_arrival = None
        self.last_error = ''
        self.received_count = 0
        self.metadata = {}
        self.control = {'mode': 'connecting', 'hold_targets_deg': None}
        self.mode_pending = True
        self.zero_force.blockSignals(True)
        self.zero_force.setChecked(True)
        self.zero_force.blockSignals(False)
        self.zero_force.setEnabled(False)
        self.zero_force.setText('Zero-force pending…')
        self.force_label.setText('Connecting; zero-force will be applied to all 21 joints.')
        self.reader = FeedbackReader(port, int(self.baud.currentText()), self.device_id.value(),
                                     rate=self.args.rate, demo=demo, side=self.model.side)
        self.connect_button.setEnabled(False)
        self.demo_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        for field in (self.port, self.baud, self.device_id):
            field.setEnabled(False)
        self.badge.setText('CONNECTING')
        self.status.setText('Opening feedback stream…')
        self.reader.start()

    def disconnect_reader(self):
        if self.reader:
            self.zero_force.setEnabled(False)
            self.reader.stop()
            self.disconnect_button.setEnabled(False)
            self.badge.setText('DISCONNECTING')

    def request_force_mode(self, enabled):
        if self.reader is None:
            return
        self.mode_pending = True
        self.zero_force.setEnabled(False)
        self.zero_force.setText('Switching…')
        self.force_label.setText('Releasing all joints…' if enabled else 'Reading current angles, then engaging position hold…')
        self.reader.request_mode(enabled)

    def show_force_mode(self, control):
        self.control = dict(control)
        self.mode_pending = False
        mode = control['mode']
        self.zero_force.blockSignals(True)
        self.zero_force.setChecked(mode == 'zero_force')
        self.zero_force.blockSignals(False)
        self.zero_force.setText({'zero_force': 'Zero-force ON', 'hold': 'Zero-force OFF · Hold'}.get(mode, 'Mode unknown'))
        self.force_label.setText({
            'zero_force': 'All 21 joints: 0 mA target. Move the fingers, then switch OFF to hold that pose.',
            'hold': 'Normal force: holding the freshly measured pose. Switch ON to move freely again.',
        }.get(mode, 'The hardware mode could not be confirmed.'))

    def tick(self):
        if self.reader:
            while True:
                try:
                    kind, payload = self.reader.events.get_nowait()
                except queue.Empty:
                    break
                if kind == 'connected':
                    try:
                        model_side = getattr(self.args, 'force_side', None) or payload['side']
                        self.load_model(model_side)
                    except Exception as exc:
                        self.last_error = f'Model load failed: {exc}'
                        self.disconnect_reader()
                        continue
                    self.metadata = payload
                    self.metadata['model_side'] = model_side
                    self.metadata['model_side_override'] = getattr(self.args, 'force_side', None)
                    side_label = model_side.upper()
                    if model_side != payload['side']:
                        side_label += f" (override; device reports {payload['side'].upper()})"
                    self.identity.setText(f"{side_label} · {payload['serial']} · firmware {payload['firmware']}"
                                          f" · SDK {payload['sdk']}")
                elif kind == 'mode':
                    self.show_force_mode(payload)
                elif kind == 'error':
                    self.last_error = payload
                    self.status.setText(payload)
                elif kind == 'closed':
                    self.reader = None
                    self.zero_force.setEnabled(False)
                    self.mode_pending = False
                    last_mode = self.control['mode']
                    self.force_label.setText(f'Disconnected · last acknowledged mode: {last_mode}. Hardware mode is not monitored.')
                    self.connect_button.setEnabled(True)
                    self.demo_button.setEnabled(True)
                    self.disconnect_button.setEnabled(False)
                    for field in (self.port, self.baud, self.device_id):
                        field.setEnabled(True)
                    self.badge.setText('DISCONNECTED')
                    self.status.setText(self.last_error or 'Disconnected. Last displayed pose is retained.')
                    break
            if self.reader and self.metadata and not self.last_error:
                try:
                    frame = self.reader.frames.get_nowait()
                except queue.Empty:
                    frame = None
                if frame:
                    self.last_arrival = frame['received_monotonic']
                    self.received_count += 1
                    if not self.freeze.isChecked():
                        self.displayed = frame
                        self.redraw()
        if self.reader and self.last_arrival is not None and not self.last_error:
            age = time.monotonic() - self.last_arrival
            frozen = self.freeze.isChecked()
            label = 'STALE' if age > .5 else ('DEMO · SIMULATED' if self.reader.demo else 'LIVE · RS485')
            if frozen:
                label += ' · FROZEN VIEW'
            if self.preview.isChecked():
                label += ' · PREVIEW'
            self.badge.setText(label)
            pose_age = time.monotonic() - self.displayed['received_monotonic'] if self.displayed else 0
            self.status.setText(f'Feedback age {age * 1000:.0f} ms · displayed pose age {pose_age:.2f} s · '
                                f'{self.received_count} frames received · '
                                + ('Stream stale; retaining last pose.' if age > .5 else 'Select a joint row to inspect its rotation axis.'))
        self.capture_button.setEnabled(self.displayed is not None)
        self.pin_button.setEnabled(self.displayed is not None)
        connected = self.reader is not None and bool(self.metadata) and not self.last_error
        fresh = self.last_arrival is not None and time.monotonic() - self.last_arrival < .5
        # Release remains available if feedback goes stale during hold.
        self.zero_force.setEnabled(connected and not self.mode_pending and
                                   (fresh or self.control['mode'] == 'hold'))

    def capture(self):
        if self.displayed is None:
            return
        try:
            folder = ROOT / 'captures'
            folder.mkdir(exist_ok=True)
            name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            target = folder / name
            report = dict(device=self.metadata, frame=self.displayed, control_at_capture=self.control,
                          captured_utc=datetime.now(timezone.utc).isoformat(),
                          frame_age_seconds=time.monotonic() - self.displayed['received_monotonic'],
                          frozen=self.freeze.isChecked(), reference=self.reference,
                          urdf=str(self.model.path.relative_to(ROOT)),
                          urdf_sha256=hashlib.sha256(self.model.path.read_bytes()).hexdigest(),
                          mapping={f'M{i}': n for i, n in enumerate(self.model.names)},
                          preview_enabled=self.preview.isChecked(), signs=self.signs.tolist(), offsets_deg=self.offsets.tolist(),
                          displayed_pose_rad={n: float(np.deg2rad(q)) for n, q in zip(self.model.names, self.model_degrees())})
            target.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
            # Finish layout and GL painting without processing another feedback frame.
            self.centralWidget().layout().activate()
            self.view.repaint()
            self.repaint()
            if not self.grab().save(str(target.with_suffix('.png'))):
                raise RuntimeError('Could not save screenshot')
            self.status.setText(f'Saved {target.name}.json and .png in {folder}')
            self.capture_button.setText('Snapshot saved ✓')
            QtCore.QTimer.singleShot(2500, lambda: self.capture_button.setText('Save snapshot'))
        except Exception as exc:
            W.QMessageBox.critical(self, 'Snapshot failed', str(exc))

    def smoke_test(self):
        try:
            assert self.displayed is not None, self.last_error or 'No feedback received'
            assert self.received_count > 5
            assert len(self.mesh_items) > 20
            self.pin_reference()
            self.freeze.setChecked(True)
            self.tick()
            self.capture()
            print(f"GUI_CHECK_OK side={self.model.side} frames={self.received_count} source={self.displayed['source']}", flush=True)
        except Exception as exc:
            print(f'GUI_CHECK_FAILED: {exc}', file=sys.stderr, flush=True)
            self.args.exit_code = 1
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.reader:
            self.reader.stop()
            self.reader.thread.join(timeout=6)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--slave-id', type=int, default=126)
    parser.add_argument('--baudrate', type=int, default=5000000, choices=[1000000, 2000000, 3000000, 5000000])
    parser.add_argument('--rate', type=int, default=30)
    parser.add_argument('--side', choices=['left', 'right'], default='left', help='Initial/demo model; hardware side is detected')
    parser.add_argument('--force-side', choices=['left', 'right'], help='Override model side when device hand-type metadata is incorrect; no hardware write')
    parser.add_argument('--connect', action='store_true', help='Connect to hardware on startup')
    parser.add_argument('--demo', action='store_true', help='Simulated feedback; never opens hardware')
    parser.add_argument('--smoke-test', action='store_true', help='Check GUI, capture a frame and exit after 4 seconds')
    args = parser.parse_args()
    if not 1 <= args.slave_id <= 247 or not 1 <= args.rate <= 100:
        parser.error('Device ID must be 1..247 and rate must be 1..100')
    if args.demo and args.connect:
        parser.error('Choose either --demo or --connect')
    app = W.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window = Viewer(args)
    window.show()
    result = app.exec()
    return getattr(args, 'exit_code', result)


if __name__ == '__main__':
    raise SystemExit(main())
