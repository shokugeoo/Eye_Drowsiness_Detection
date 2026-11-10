#!/usr/bin/env python3

import argparse
import struct
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import serial

WIDTH = 160
HEIGHT = 120
PIXELS = WIDTH * HEIGHT
BAUD = 921600

LEFT_EYE = (65, 58)
RIGHT_EYE = (88, 58)
ROI_SIZE = 16

MAGIC = b"EY24"
PROTOCOL_VERSION = 2
HEADER_SIZE = 19
PACKET_SIZE = HEADER_SIZE + PIXELS + 4


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.bad_versions = 0

    def feed(self, data):
        self.buffer.extend(data)
        frames = []

        while True:
            position = self.buffer.find(MAGIC)

            if position < 0:
                # Retain the last three bytes for a split magic marker.
                if len(self.buffer) > 3:
                    del self.buffer[:-3]
                break

            if position > 0:
                del self.buffer[:position]

            if len(self.buffer) < HEADER_SIZE:
                break

            if self.buffer[4] != PROTOCOL_VERSION:
                self.bad_versions += 1
                del self.buffer[0]
                continue

            if len(self.buffer) < PACKET_SIZE:
                break

            packet = self.buffer[:PACKET_SIZE]

            payload = packet[HEADER_SIZE:HEADER_SIZE + PIXELS]
            received_crc = struct.unpack_from(
                "<I", packet, HEADER_SIZE + PIXELS
            )[0]

            calculated_crc = zlib.crc32(packet[4:HEADER_SIZE])
            calculated_crc = zlib.crc32(
                payload, calculated_crc
            ) & 0xFFFFFFFF

            if calculated_crc != received_crc:
                self.crc_errors += 1
                del self.buffer[0]
                continue

            scores = struct.unpack("4b", packet[5:9])
            frame_number = struct.unpack("<I", packet[9:13])[0]

            left_perclos_x10, right_perclos_x10, coverage_ms = (
                struct.unpack_from("<HHH", packet, 13)
            )

            image = np.frombuffer(
                payload, dtype=np.uint8
            ).reshape((HEIGHT, WIDTH)).copy()

            frames.append((
                frame_number,
                scores,
                image,
                left_perclos_x10,
                right_perclos_x10,
                coverage_ms,
            ))
            del self.buffer[:PACKET_SIZE]

        return frames


def eye_state(closed_score, open_score):
    return "CLOSED" if closed_score >= open_score else "OPEN"


def make_preview(
    frame_number,
    scores,
    gray,
    clahe,
    left_perclos_x10,
    right_perclos_x10,
    coverage_ms,
):
    lc, lo, rc, ro = scores

    # CLAHE is display-only. The CNN receives its original preprocessing.
    display_gray = clahe.apply(gray)

    # Rotate the preview 180 degrees for upright viewing.
    display_gray = cv2.rotate(display_gray, cv2.ROTATE_180)
    canvas = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)

    scale = 4
    canvas = cv2.resize(
        canvas,
        (WIDTH * scale, HEIGHT * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    for x, y in (LEFT_EYE, RIGHT_EYE):
        # Transform raw camera coordinates to the rotated preview.
        display_x = WIDTH - (x + ROI_SIZE)
        display_y = HEIGHT - (y + ROI_SIZE)

        cv2.rectangle(
            canvas,
            (display_x * scale, display_y * scale),
            ((display_x + ROI_SIZE) * scale,
             (display_y + ROI_SIZE) * scale),
            (0, 255, 0),
            2,
        )

    left_label = f"LEFT: {eye_state(lc, lo)}  C={lc} O={lo}"
    right_label = f"RIGHT: {eye_state(rc, ro)}  C={rc} O={ro}"

    average_perclos_x10 = (
        left_perclos_x10 + right_perclos_x10
    ) / 2.0

    perclos_label = (
        f"PERCLOS L={left_perclos_x10 / 10.0:.1f}%  "
        f"R={right_perclos_x10 / 10.0:.1f}%  "
        f"AVG={average_perclos_x10 / 10.0:.1f}%"
    )

    # Drowsiness status based on average PERCLOS.
    if average_perclos_x10 >= 300:
        status_label = "STATUS: DROWSY!"
        status_color = (0, 0, 255)       # Red
    elif average_perclos_x10 >= 200:
        status_label = "STATUS: WARNING"
        status_color = (0, 255, 255)     # Yellow
    else:
        status_label = "STATUS: NORMAL"
        status_color = (80, 255, 100)    # Green

    coverage_label = (
        f"VALID DATA (60 completed seconds): {coverage_ms / 1000.0:.1f}s / 60.0s"
    )

    # Six text rows fit above the calibrated eye boxes.
    cv2.rectangle(canvas, (0, 0), (640, 175), (25, 25, 25), -1)

    rows = [
        (f"FRAME {frame_number}", 22, (255, 255, 255)),
        (left_label, 48, (80, 255, 100)),
        (right_label, 73, (80, 255, 100)),
        (perclos_label, 98, (100, 220, 255)),
        (coverage_label, 123, (220, 220, 220)),
        (status_label, 151, status_color),
    ]

    for text, y, color in rows:
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas

def main():
    parser = argparse.ArgumentParser(
        description="MAX78000 EYE24 live camera viewer"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--output", default="captures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_parser = FrameParser()
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    latest = None
    window = "EYE24 - MAX78000 Live Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 760)

    print(f"Opening {args.port} at {args.baud} baud...")
    print("Waiting for MAX78000 frame packets...")
    print("S = save frame | Q = quit")

    try:
        with serial.Serial(
            args.port,
            args.baud,
            timeout=0.05,
            write_timeout=0.2,
        ) as ser:

            # Discard stale bytes; wait for the next complete packet.
            ser.reset_input_buffer()

            while True:
                data = ser.read(4096)

                if data:
                    frames = packet_parser.feed(data)

                    for frame in frames:
                        latest = frame
                        (
                            number,
                            scores,
                            _,
                            left_perclos_x10,
                            right_perclos_x10,
                            coverage_ms,
                        ) = frame

                        lc, lo, rc, ro = scores
                        average_perclos_x10 = (
                            left_perclos_x10 + right_perclos_x10
                        ) / 2.0

                        print(
                            f"Frame {number} | "
                            f"LEFT {eye_state(lc, lo)} "
                            f"(C={lc}, O={lo}) | "
                            f"RIGHT {eye_state(rc, ro)} "
                            f"(C={rc}, O={ro}) | "
                            f"PERCLOS L={left_perclos_x10 / 10.0:.1f}% "
                            f"R={right_perclos_x10 / 10.0:.1f}% "
                            f"AVG={average_perclos_x10 / 10.0:.1f}% | "
                            f"coverage={coverage_ms / 1000.0:.1f}/60.0s"
                        )

                if latest is not None:
                    (
                        number,
                        scores,
                        gray,
                        left_perclos_x10,
                        right_perclos_x10,
                        coverage_ms,
                    ) = latest

                    preview = make_preview(
                        number,
                        scores,
                        gray,
                        clahe,
                        left_perclos_x10,
                        right_perclos_x10,
                        coverage_ms,
                    )
                else:
                    preview = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        preview,
                        "WAITING FOR EYE24 FRAME...",
                        (55, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key in (ord("s"), ord("S")) and latest is not None:
                    (
                        number,
                        scores,
                        gray,
                        left_perclos_x10,
                        right_perclos_x10,
                        coverage_ms,
                    ) = latest

                    annotated = make_preview(
                        number,
                        scores,
                        gray,
                        clahe,
                        left_perclos_x10,
                        right_perclos_x10,
                        coverage_ms,
                    )

                    annotated_path = (
                        output_dir / f"frame_{number:06d}_annotated.png"
                    )
                    gray_path = (
                        output_dir / f"frame_{number:06d}_gray.png"
                    )

                    cv2.imwrite(str(annotated_path), annotated)
                    cv2.imwrite(str(gray_path), gray)

                    print(f"Saved: {annotated_path}")
                    print(f"Saved: {gray_path}")

                elif key in (ord("s"), ord("S")):
                    print("No camera frame received yet.")

    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        cv2.destroyAllWindows()

    print(
        f"CRC errors: {packet_parser.crc_errors} | "
        f"Bad protocol versions: {packet_parser.bad_versions}"
    )


if __name__ == "__main__":
    main()
