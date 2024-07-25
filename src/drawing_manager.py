from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon


class DrawingManager:
    def __init__(self):
        """
        Initialise the DrawingManager class.
        """
        # Load the font used for drawing labels on the image
        self.font = ImageFont.truetype(
            'assets/fonts/NotoSansTC-VariableFont_wght.ttf', 20,
        )

        # Mapping of category IDs to their corresponding names
        self.category_id_to_name = {
            0: '安全帽',
            1: '口罩',
            2: '無安全帽',
            3: '無口罩',
            4: '無安全背心',
            5: '人員',
            6: '安全錐',
            7: '安全背心',
            8: '機具',
            9: '車輛',
        }

        # Define colours for each category
        self.colors = {
            '安全帽': (0, 255, 0),
            '安全背心': (0, 255, 0),
            '機具': (255, 225, 0),
            '車輛': (255, 255, 0),
            '無安全帽': (255, 0, 0),
            '無安全背心': (255, 0, 0),
            '人員': (255, 165, 0),
        }

        # Generate exclude_labels automatically
        self.exclude_labels = [
            label for label in self.category_id_to_name.values()
            if label not in self.colors
        ]

    def draw_safety_cones_polygon(
        self,
        frame: np.ndarray,
        datas: list[list[float]],
    ) -> tuple[np.ndarray, Polygon | None]:
        """
        Draws safety cones on the given frame and forms a polygon from them.

        Args:
            frame (np.ndarray): The frame on which to draw safety cones.
            datas (List[List[float]]): The detection data.

        Returns:
            Tuple[np.ndarray, Optional[Polygon]]: The frame with
                safety cones drawn, and the polygon formed by the safety cones.
        """
        if not datas:
            return frame, None

        # Get positions of safety cones
        cone_positions = np.array([
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas if data[5] == 6
        ])

        # Check if there are at least three safety cones to form a polygon
        if len(cone_positions) < 3:
            return frame, None

        # Create a polygon from the cone positions
        polygon = MultiPoint(cone_positions).convex_hull

        if isinstance(polygon, Polygon):
            polygon_points = np.array(
                polygon.exterior.coords, dtype=np.float32,
            )

            # Convert frame to PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Create a transparent layer
            overlay = Image.new('RGBA', frame_pil.size, (255, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            # Draw the polygon with semi-transparent pink colour
            overlay_draw.polygon(
                [tuple(point) for point in polygon_points],
                fill=(255, 105, 180, 128),
            )

            # Composite the overlay with the original image
            frame_pil = Image.alpha_composite(
                frame_pil.convert('RGBA'), overlay,
            )

            # Convert back to OpenCV image
            frame = cv2.cvtColor(
                np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR,
            )

            # Draw the polygon border
            cv2.polylines(
                frame, [
                    polygon_points.astype(
                        np.int32,
                    ),
                ], isClosed=True, color=(255, 0, 255), thickness=2,
            )
        else:
            print('Warning: Convex hull is not a polygon.')

        return frame, polygon

    def draw_detections_on_frame(
        self,
        frame: np.ndarray,
        datas: list[list[float]],
    ) -> tuple[np.ndarray, Polygon | None]:
        """
        Draws detections on the given frame.

        Args:
            frame (np.ndarray): The frame on which to draw detections.
            datas (List[List[float]]): The detection data.

        Returns:
            Tuple[np.ndarray, Optional[Polygon]]: The frame with
                detections drawn, and the polygon formed by the safety cones.
        """
        # Draw safety cones first
        frame, polygon = self.draw_safety_cones_polygon(frame, datas)

        # Convert the frame to RGB and create a PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        for data in datas:
            x1, y1, x2, y2, _, label_id = data
            label_id = int(label_id)
            if label_id in self.category_id_to_name:
                label = self.category_id_to_name[label_id]
            else:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if label not in self.exclude_labels:
                color = self.colors.get(label, (255, 255, 255))
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                text = f"{label}"
                text_bbox = draw.textbbox((x1, y1), text, font=self.font)
                text_width, text_height = text_bbox[2] - \
                    text_bbox[0], text_bbox[3] - text_bbox[1]
                text_background = (
                    x1, y1 - text_height -
                    5, x1 + text_width, y1,
                )
                draw.rectangle(text_background, fill=color)
                draw.text(
                    (x1, y1 - text_height - 5), text,
                    fill=(0, 0, 0), font=self.font,
                )

        # Convert the PIL image back to OpenCV format
        frame_with_detections = cv2.cvtColor(
            np.array(pil_image), cv2.COLOR_RGB2BGR,
        )
        return frame_with_detections, polygon

    def save_frame(self, frame_bytes: bytearray, output_filename: str) -> None:
        """
        Saves detected frame to given output folder and filename.

        Args:
            frame_bytes (bytearray): The byte stream of the frame.
            output_filename (str): The output filename.
        """
        # Create the output directory if it does not exist
        output_dir = Path('detected_frames')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the output path
        output_path = output_dir / f"{output_filename}.png"

        # Save the byte stream to the output path
        with open(output_path, 'wb') as f:
            f.write(frame_bytes)

        # Clean up
        del output_dir, output_path, frame_bytes
        gc.collect()


if __name__ == '__main__':
    # Example usage (replace with actual usage)
    # Load frame and detection data (example)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Example data including objects and safety cones
    datas = [
        # Example objects (安全帽, 人員, 車輛)
        [50, 50, 150, 150, 0.95, 0],   # 安全帽
        [200, 200, 300, 300, 0.85, 5],  # 人員
        [400, 400, 500, 500, 0.75, 9],  # 車輛

        # Example safety cones (安全錐)
        [100, 100, 120, 120, 0.9, 6],
        [250, 250, 270, 270, 0.8, 6],
        [450, 450, 470, 470, 0.7, 6],
        [500, 200, 520, 220, 0.7, 6],
        [150, 400, 170, 420, 0.7, 6],
    ]

    # Initialise DrawingManager class
    drawer_saver = DrawingManager()

    # Draw detections on frame (including safety cones)
    frame_with_detections, polygon = drawer_saver.draw_detections_on_frame(
        frame, datas,
    )

    # Save the frame with detections
    output_filename = 'frame_001'
    frame_bytes = cv2.imencode('.png', frame_with_detections)[1].tobytes()
    drawer_saver.save_frame(frame_bytes, output_filename)
