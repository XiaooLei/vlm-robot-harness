"""ZED acquisition: three views, optional RGB recording, JPEG+blob feedback."""
import base64
import json
from pathlib import Path
import threading
import time

class ThreeCameras:
    """Three physical ZED cameras, left image from each, kept open in this loop."""
    def __init__(self, serials, attempts=3):
        import pyzed.sl as sl
        self.sl, self.cameras, self.last_stamp = sl, [], {}
        self.io_lock = threading.Lock()
        self.record_thread = None
        self.record_stop = threading.Event()
        try:
            for role, serial in zip(("wrist", "exterior_1", "exterior_2"), serials):
                # A freshly released ZED sometimes fails to reopen; retry rather
                # than aborting a whole run on a transient open failure.
                for attempt in range(attempts):
                    cam = sl.Camera()
                    init = sl.InitParameters()
                    init.set_from_serial_number(int(serial))
                    init.camera_resolution = sl.RESOLUTION.HD720
                    init.camera_fps = 15
                    init.depth_mode = sl.DEPTH_MODE.NONE
                    status = cam.open(init)
                    if status == sl.ERROR_CODE.SUCCESS:
                        self.cameras.append((role, serial, cam))
                        break
                    cam.close()
                    if attempt + 1 < attempts:
                        time.sleep(1.)
                if status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"Camera {role}/{serial}: {status} after {attempts} attempts; no camera recovery attempted")
        except BaseException:
            self.close()
            raise

    def read_frame(self, cam, discard=1):
        # Recorder and observations share SDK handles, never concurrent grabs.
        with self.io_lock:
            for _ in range(discard):
                if cam.grab() != self.sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError("Camera grab failed")
            stamp = cam.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            mat = self.sl.Mat()
            if cam.retrieve_image(mat, self.sl.VIEW.LEFT) != self.sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Camera image retrieval failed")
            return stamp, mat.get_data()[:, :, :3].copy()

    def start_recording(self, directory, fps=5):
        import cv2
        if self.record_thread is not None:
            raise RuntimeError("Recording already active")
        directory.mkdir(parents=True, exist_ok=True)
        writers = {}
        try:
            for role, serial, _ in self.cameras:
                writer = cv2.VideoWriter(str(directory / f"{role}_{serial}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
                writers[role] = writer
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {role}")
        except Exception:
            for writer in writers.values(): writer.release()
            raise
        self.record_stop.clear()
        def record_loop():
            counts = {role: 0 for role in writers}
            error = None
            started = time.time()
            try:
                with (directory / "frames.jsonl").open("w") as log:
                    while not self.record_stop.is_set():
                        tick = time.monotonic()
                        for role, serial, cam in self.cameras:
                            stamp, frame = self.read_frame(cam)
                            writers[role].write(cv2.resize(frame, (960, 540)))
                            log.write(json.dumps(dict(role=role, frame=counts[role],
                                camera_timestamp_ns=stamp, time=time.time())) + "\n")
                            counts[role] += 1
                        log.flush()
                        self.record_stop.wait(max(0, 1/fps - (time.monotonic()-tick)))
            except Exception as err:
                error = str(err)
                print(json.dumps(dict(recording_error=error)), flush=True)
            finally:
                for writer in writers.values(): writer.release()
                (directory / "manifest.json").write_text(json.dumps(dict(
                    started=started, ended=time.time(), fps=fps, frames=counts,
                    error=error, depth=False,
                    timing="Nominal 5 fps; actual frame timestamps are in frames.jsonl")))
        self.record_thread = threading.Thread(target=record_loop, name="rgb-recorder", daemon=True)
        self.record_thread.start()

    def capture(self, directory):
        import cv2
        result = []
        for role, serial, cam in self.cameras:
            stamp, image = self.read_frame(cam, discard=3)
            if stamp <= self.last_stamp.get(role, 0):
                raise RuntimeError(f"Stale camera image: {role}")
            self.last_stamp[role] = stamp
            image = cv2.resize(image, (960, 540))
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            path = directory / f"{role}_{serial}.jpg"
            path.write_bytes(jpeg.tobytes())
            entry = dict(role=role, serial=serial, path=str(path),
                         camera_timestamp_ns=stamp, captured_monotonic=time.monotonic(),
                         url="data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            if role == "wrist":
                # Numeric feedback for the model: where the yellow object sits in
                # the wrist image, so successive steps calibrate image->base.
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (20, 130, 100), (40, 255, 255))
                count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
                found = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 400]
                if found:
                    best = max(found, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                    entry["yellow_uv"] = [round(float(centers[best][0]) / image.shape[1], 3),
                                          round(float(centers[best][1]) / image.shape[0], 3)]
                    entry["yellow_area_px"] = int(stats[best, cv2.CC_STAT_AREA])
                else:
                    entry["yellow_uv"] = None
            result.append(entry)
        return result

    def close(self):
        self.record_stop.set()
        if self.record_thread is not None:
            self.record_thread.join()
            self.record_thread = None
        for _, _, cam in self.cameras:
            cam.close()
        self.cameras.clear()
