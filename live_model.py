"""URDF kinematics and explicit SDK motor mapping for the live viewer."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# SDK motor order M0..M20; thumb Rot/MCP/IP map to URDF MCP/PIP/DIP.
JOINT_LABELS = ('little_MPR', 'little_MCP', 'little_PIP', 'little_DIP', 'ring_MPR', 'ring_MCP', 'ring_PIP', 'ring_DIP', 'middle_MPR', 'middle_MCP', 'middle_PIP', 'middle_DIP', 'index_MPR', 'index_MCP', 'index_PIP', 'index_DIP', 'thumb_MCP', 'thumb_PIP', 'thumb_DIP', 'thumb_CMP', 'thumb_CMR')

ROOT = Path(__file__).resolve().parent
DISPLAY_ORDER = (19, 20, 16, 17, 18, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        roll, pitch, yaw = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        result[:3, :3] = (rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch)
                         @ rotation([1, 0, 0], roll))
    return result


class HandModel:
    def __init__(self, side):
        if side not in ('left', 'right'):
            raise ValueError('Unknown hand side')
        self.side = side
        self.path = ROOT / f'revo3_system/urdf/revo3_{side}.urdf'
        root = ET.parse(self.path).getroot()
        self.names = [f'{side}_{label}_joint' for label in JOINT_LABELS]
        self.joints = []
        self.by_name = {}
        pending = list(root.findall('joint'))
        children = {j.find('child').get('link') for j in pending}
        self.roots = {l.get('name') for l in root.findall('link')} - children
        visited = set(self.roots)
        while pending:
            progressed = False
            for element in pending[:]:
                parent = element.find('parent').get('link')
                if parent not in visited:
                    continue
                name = element.get('name')
                child = element.find('child').get('link')
                axis = element.find('axis')
                limit = element.find('limit')
                kind = element.get('type')
                if kind not in ('fixed', 'revolute'):
                    raise ValueError(f'Unsupported joint type: {kind}')
                joint = dict(name=name, parent=parent, child=child, origin=origin(element.find('origin')),
                             axis=np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None,
                             motor=self.names.index(name) if kind != 'fixed' else None,
                             limits=(float(limit.get('lower')), float(limit.get('upper'))) if limit is not None else None)
                self.joints.append(joint)
                self.by_name[name] = joint
                visited.add(child)
                pending.remove(element)
                progressed = True
            if not progressed:
                raise ValueError('Disconnected URDF joint graph')
        self.visuals = []
        materials = {m.get('name'): np.fromstring(m.find('color').get('rgba'), sep=' ')
                     for m in root.findall('material') if m.find('color') is not None}
        for link in root.findall('link'):
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    continue
                mat = visual.find('material')
                color = np.array([0.65, 0.72, 0.8, 1.0])
                if mat is not None:
                    color = materials.get(mat.get('name'), color)
                    if mat.find('color') is not None:
                        color = np.fromstring(mat.find('color').get('rgba'), sep=' ')
                self.visuals.append(dict(link=link.get('name'), origin=origin(visual.find('origin')),
                                         path=self.path.parent / mesh.get('filename'), color=color,
                                         scale=np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')))
        if set(self.names) - set(self.by_name):
            raise ValueError('URDF does not contain the expected 21 joints')

    def forward(self, degrees):
        degrees = np.asarray(degrees, dtype=float)
        if degrees.shape != (21,) or not np.isfinite(degrees).all():
            raise ValueError('Expected 21 finite joint angles')
        poses = {name: np.eye(4) for name in self.roots}
        axes = {}
        for j in self.joints:
            transform = poses[j['parent']] @ j['origin']
            if j['motor'] is not None:
                axes[j['motor']] = (transform[:3, 3].copy(), transform[:3, :3] @ j['axis'])
                turn = np.eye(4)
                turn[:3, :3] = rotation(j['axis'], np.deg2rad(degrees[j['motor']]))
                transform = transform @ turn
            poses[j['child']] = transform
        return poses, axes

    def limits_deg(self, motor):
        return np.rad2deg(self.by_name[self.names[motor]]['limits'])


def validate_frame(status):
    frame = {}
    for field in ('positions', 'velocities', 'currents', 'statuses', 'errors'):
        values = list(getattr(status, field))
        if len(values) != 21 or not np.isfinite(values).all():
            raise ValueError(f'Invalid {field}: expected 21 finite values')
        frame[field] = values
    return frame
