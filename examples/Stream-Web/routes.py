from flask import Flask, render_template, abort, make_response, Response
from flask_limiter import Limiter
from .utils import get_labels, get_image_data

def register_routes(app: Flask, limiter: Limiter, r) -> None:
    @app.route('/')
    @limiter.limit("60 per minute")
    def index() -> str:
        """
        Serve the index page with a dynamically generated list of labels from Redis.

        Returns:
            str: The rendered 'index.html' page.
        """
        labels = get_labels(r)
        return render_template('index.html', labels=labels)

    @app.route('/label/<label>', methods=['GET'])
    @limiter.limit("60 per minute")
    def label_page(label: str) -> str:
        """
        Serve a page for each label, displaying images under that label.

        Args:
            label (str): The label/category of the images.

        Returns:
            str: The rendered 'label.html' page for the specific label.
        """
        image_data = get_image_data(r, label)
        return render_template('label.html', label=label, image_data=image_data)

    @app.route('/image/<label>/<filename>.png')
    @limiter.limit("60 per minute")
    def image(label: str, filename: str) -> Response:
        """
        Serve an image file from Redis.

        Args:
            label (str): The label/category of the image.
            filename (str): The filename of the image.

        Returns:
            Response: The image file as a response.
        """
        redis_key = f"{label}_{filename}"
        img_encoded = r.get(redis_key)

        if img_encoded is None:
            abort(404, description="Resource not found")

        response = make_response(img_encoded)
        response.headers.set('Content-Type', 'image/png')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response

    @app.route('/camera/<label>/<camera_id>')
    @limiter.limit("60 per minute")
    def camera_page(label: str, camera_id: str) -> str:
        """
        Serve a page for viewing the camera stream.

        Args:
            label (str): The label/category of the camera.
            camera_id (str): The specific ID of the camera to view.

        Returns:
            str: The rendered 'camera.html' page for the specific camera.
        """
        return render_template('camera.html', label=label, camera_id=camera_id)
