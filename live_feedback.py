"""Single-owner RS485 feedback and zero-force/current-pose hold control."""
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import queue
import threading
import time

import numpy as np

from live_model import validate_frame


class FeedbackReader:
    def __init__(self, port, baudrate, slave_id, rate=30, demo=False, side='left'):
        self.port, self.baudrate, self.slave_id = port, baudrate, slave_id
        self.rate, self.demo, self.side = rate, demo, side
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.commands = queue.Queue()
        self.control = {'mode': 'disconnected', 'hold_targets_deg': None}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='Revo3-feedback')

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def request_mode(self, zero_force):
        self.commands.put('zero_force' if zero_force else 'hold')

    def announce_mode(self, mode, targets=None):
        self.control = dict(mode=mode, hold_targets_deg=targets,
                            acknowledged_utc=datetime.now(timezone.utc).isoformat())
        self.events.put(('mode', dict(self.control)))
        print(f"CONTROL {'DEMO' if self.demo else self.port}: {mode}", flush=True)

    async def change_mode(self, ctx, mode):
        """Hold uses a NEW device read, never GUI angles, corrections, or a cached pose."""
        zeros = [0.0] * 21
        try:
            targets = None
            if mode == 'hold':
                status = await asyncio.wait_for(ctx.revo3_get_motor_status_data(self.slave_id), timeout=2.0)
                targets = validate_frame(status)['positions']
                if self.stop_event.is_set():
                    return
                # Mode 0 is normal firmware position control. Capture and engage once.
                await asyncio.wait_for(ctx.revo3_multi_joint_control(self.slave_id, 0, targets), timeout=2.0)
            elif mode == 'zero_force':
                # Mode 2 is current control. Zero current releases all five fingers.
                await asyncio.wait_for(ctx.revo3_multi_joint_control(self.slave_id, 2, zeros), timeout=2.0)
            else:
                raise ValueError(f'Unknown control mode: {mode}')
            self.announce_mode(mode, targets)
        except Exception as exc:
            # An interrupted mode write can have reached the device. Try releasing it.
            try:
                await asyncio.wait_for(ctx.revo3_multi_joint_control(self.slave_id, 2, zeros), timeout=2.0)
                self.announce_mode('zero_force')
                outcome = 'Returned to zero-force.'
            except Exception:
                self.announce_mode('unknown')
                outcome = 'Hardware mode could not be confirmed.'
            raise RuntimeError(f'Mode change failed: {exc}. {outcome}') from exc

    def publish(self, frame):
        frame['received_monotonic'] = time.monotonic()
        frame['received_utc'] = datetime.now(timezone.utc).isoformat()
        frame['source'] = 'demo' if self.demo else 'hardware'
        frame['control'] = dict(self.control)
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(frame)

    def run(self):
        try:
            if self.demo:
                self.run_demo()
            else:
                asyncio.run(self.run_hardware())
        except Exception as exc:
            self.events.put(('error', f'{type(exc).__name__}: {exc}'))
        finally:
            self.events.put(('closed', None))

    def run_demo(self):
        self.events.put(('connected', dict(side=self.side, serial='SIMULATED', firmware='demo',
                                           sdk='none', port='none', slave_id=None, teaching_mode=None)))
        self.announce_mode('zero_force')
        start = time.monotonic()
        held = None
        while not self.stop_event.is_set():
            t = time.monotonic() - start
            q = np.zeros(21)
            for finger in range(4):
                q[finger * 4 + 1:finger * 4 + 4] = (1 - np.cos(t * 1.2 + finger * .5)) * np.array([20, 25, 15])
            q[16:21] = (1 - np.cos(t * .7)) * np.array([15, 12, 10, 25, 20])
            try:
                mode = self.commands.get_nowait()
                held = q.copy() if mode == 'hold' else None
                self.announce_mode(mode, held.tolist() if held is not None else None)
            except queue.Empty:
                pass
            if held is not None:
                q = held
            self.publish(dict(positions=q.tolist(), velocities=[0.] * 21, currents=[0.] * 21,
                              statuses=[0] * 21, errors=[0] * 21))
            self.stop_event.wait(1 / self.rate)

    async def run_hardware(self):
        from bc_revo3_sdk import main_mod as sdk
        rates = {1000000: sdk.Baudrate.Baud1Mbps, 2000000: sdk.Baudrate.Baud2Mbps,
                 3000000: sdk.Baudrate.Baud3Mbps, 5000000: sdk.Baudrate.Baud5Mbps}
        ctx = None

        async def read(method):
            return await asyncio.wait_for(method(self.slave_id), timeout=2.0)

        try:
            ctx = await asyncio.wait_for(sdk.modbus_open(self.port, rates[self.baudrate]), timeout=3.0)
            info = await read(ctx.revo3_get_device_info)
            hand_type = await read(ctx.revo3_get_hand_type)
            if hand_type not in (1, 2):
                raise ValueError(f'Unsupported hand type {hand_type}')
            # Validate this is a 21-joint feedback device before issuing control.
            validate_frame(await read(ctx.revo3_get_motor_status_data))
            await self.change_mode(ctx, 'zero_force')
            metadata = dict(side={1: 'right', 2: 'left'}[hand_type], serial=info.serial_number,
                            firmware=info.firmware_version, hardware=str(info.hardware_type),
                            hardware_version=str(info.hardware_version), sdk=version('bc-revo3-sdk'),
                            port=self.port, baudrate=self.baudrate, slave_id=self.slave_id)
            for key, method in [('teaching_mode', ctx.revo3_get_teaching_mode),
                                ('stored_zero_offsets_deg', ctx.revo3_get_zero_position),
                                ('device_limits_deg', ctx.revo3_get_all_joint_position_limits)]:
                try:
                    metadata[key] = await read(method)
                except Exception as exc:
                    metadata[key] = None
                    metadata[key + '_read_error'] = str(exc)
            self.events.put(('connected', metadata))
            while not self.stop_event.is_set():
                start = time.monotonic()
                try:
                    mode = self.commands.get_nowait()
                except queue.Empty:
                    mode = None
                if mode is not None:
                    await self.change_mode(ctx, mode)
                if self.stop_event.is_set():
                    break
                status = await read(ctx.revo3_get_motor_status_data)
                frame = validate_frame(status)
                frame['read_duration_ms'] = (time.monotonic() - start) * 1000
                self.publish(frame)
                await asyncio.sleep(max(0, 1 / self.rate - (time.monotonic() - start)))
        finally:
            if ctx is not None:
                result = sdk.modbus_close(ctx)
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=3.0)
