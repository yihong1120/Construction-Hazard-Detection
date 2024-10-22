from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from shapely.geometry import Polygon

from .lang_config import LANGUAGES


class DrawingManager:
    """
    A class for drawing detections on frames and saving them to disk.
    """

    # Class variable for caching default font
    default_font: ImageFont.ImageFont | None = None

    def __init__(self) -> None:
        """
        Initialise the DrawingManager class.
        """
        # Font cache to avoid repeated loading
        self.font_cache: dict[str, ImageFont.ImageFont] = {}

        # Load default font if not already loaded
        if DrawingManager.default_font is None:
            DrawingManager.default_font = ImageFont.load_default()

    def get_font(self, language: str) -> ImageFont.ImageFont:
        """
        Load the appropriate font based on the language input, with caching.

        Args:
            language (str): The language to use for the font.

        Returns:
            ImageFont.ImageFont: The loaded font object.
        """
        # Select font path based on language
        if language == 'th':
            font_path = 'assets/fonts/NotoSansThai-VariableFont_wdth.ttf'
        else:
            font_path = 'assets/fonts/NotoSansTC-VariableFont_wght.ttf'

        # Check if the font is already in the cache
        if font_path in self.font_cache:
            return self.font_cache[font_path]

        # Load and cache the font
        try:
            font = ImageFont.truetype(font_path, 20)
        except OSError:
            print(f"Error loading font from {font_path}. Using default font.")
            if DrawingManager.default_font is None:
                # Load default font only once
                DrawingManager.default_font = ImageFont.load_default()
            return DrawingManager.default_font

        # Cache the font using the path as the key
        self.font_cache[font_path] = font
        return font

    def draw_polygons(
        self,
        frame: np.ndarray,
        polygons: list[Polygon],
    ) -> np.ndarray:
        """
        Draws polygons on the given frame.

        Args:
            frame (np.ndarray): The frame on which to draw polygons.
            polygons (List[Polygon]): list of polygons containing safety cones.

        Returns:
            np.ndarray: The frame with polygons drawn.
        """
        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Create a transparent layer
        overlay = Image.new('RGBA', frame_pil.size, (255, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw the polygons
        for polygon in polygons:
            # Get polygon points
            points = polygon.exterior.coords if isinstance(
                polygon, Polygon,
            ) else polygon.coords
            points = [
                tuple(point)
                for point in points
            ]  # Convert points to tuples

            # Draw the polygon or line
            overlay_draw.polygon(points, fill=(255, 105, 180, 128))

            # Draw the polygon border
            overlay_draw.line(
                points + [points[0]],
                fill=(255, 0, 255), width=2,
            )

        # Composite the overlay with the original image
        frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), overlay)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(
            np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR,
        )

        return frame

    def draw_detections_on_frame(
        self,
        frame: np.ndarray,
        polygons: list[Polygon],
        datas: list[list[float]],
        language: str = 'en',  # Accept language as input
    ) -> np.ndarray:
        """
        Draws detections on the given frame
        and supports dynamic language selection.

        Args:
            frame (np.ndarray): The frame on which to draw detections.
            datas (List[List[float]]): The detection data.
            language (str): The language to use for labels.

        Returns:
            np.ndarray: The frame with detections drawn.
        """
        # Load language configuration
        lang_config = LANGUAGES.get(language, LANGUAGES['en'])

        # Define category names based on language
        category_id_to_name: dict[int, str] = {
            0: lang_config['helmet'],
            1: lang_config['mask'],
            2: lang_config['no_helmet'],
            3: lang_config['no_mask'],
            4: lang_config['no_vest'],
            5: lang_config['person'],
            6: lang_config['cone'],
            7: lang_config['vest'],
            8: lang_config['machinery'],
            9: lang_config['vehicle'],
        }

        # Define colours for each category
        colours: dict[str, tuple[int, int, int]] = {
            lang_config['helmet']: (0, 255, 0),
            lang_config['vest']: (0, 255, 0),
            lang_config['machinery']: (255, 225, 0),
            lang_config['vehicle']: (255, 255, 0),
            lang_config['no_helmet']: (255, 0, 0),
            lang_config['no_vest']: (255, 0, 0),
            lang_config['person']: (255, 165, 0),
        }

        # Load the font based on the input language
        font = self.get_font(language)

        # Draw polygons first
        if polygons:
            frame = self.draw_polygons(frame, polygons)

        # Convert the frame to RGB and create a PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Draw the detections on the frame
        for data in datas:
            x1, y1, x2, y2, _, label_id = data
            label_id = int(label_id)
            if label_id in category_id_to_name:
                label = category_id_to_name[label_id]
            else:
                continue

            # Draw the bounding box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if label in colours:
                colour = colours.get(label, (255, 255, 255))
                draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)
                text = f"{label}"
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width, text_height = text_bbox[2] - \
                    text_bbox[0], text_bbox[3] - text_bbox[1]
                text_background = (
                    x1, y1 - text_height -
                    5, x1 + text_width, y1,
                )
                draw.rectangle(text_background, fill=colour)
                draw.text(
                    (x1, y1 - text_height - 5), text,
                    fill=(0, 0, 0), font=font,
                )

        # Convert the PIL image back to OpenCV format
        frame_with_detections = cv2.cvtColor(
            np.array(pil_image), cv2.COLOR_RGB2BGR,
        )

        return frame_with_detections

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


def main() -> None:
    """
    Main function to process and save the frame with detections.
    """
    drawer_saver = DrawingManager()

    # Load frame and detection data (example)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Example data including objects and safety cones
    datas = [
        # Example objects (Hardhat, Person, Vehicle)
        [50, 50, 150, 150, 0.95, 0],   # Hardhat
        [200, 200, 300, 300, 0.85, 5],  # Person
        [400, 400, 500, 500, 0.75, 9],  # Vehicle

        # Example safety cones (Safety Cone)
        [100, 100, 120, 120, 0.9, 6],
        [250, 250, 270, 270, 0.8, 6],
        [450, 450, 470, 470, 0.7, 6],
        [500, 200, 520, 220, 0.7, 6],
        [150, 400, 170, 420, 0.7, 6],
    ]

    # Define the points directly
    points = [(100, 100), (250, 250), (450, 450), (500, 200), (150, 400)]
    polygon = Polygon(points).convex_hull

    # Draw detections on frame (including safety cones)
    frame_with_detections = drawer_saver.draw_detections_on_frame(
        frame, [polygon], datas, language='en',
    )

    # Save the frame with detections
    output_filename = 'frame_001'
    frame_bytes = cv2.imencode('.png', frame_with_detections)[1].tobytes()
    drawer_saver.save_frame(bytearray(frame_bytes), output_filename)


if __name__ == '__main__':
    main()
