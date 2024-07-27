from __future__ import annotations

from flask import escape
from flask import Flask
from flask import request

from .models import db
from .user_operation import add_user
from .user_operation import delete_user
from .user_operation import update_password
from .user_operation import update_username

app = Flask(__name__)

# Configuration for SQLAlchemy, replace 'your_database_uri_here'
# with your actual database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'your_database_uri_here'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create all database tables before the first run
with app.app_context():
    db.create_all()


@app.route('/add_user', methods=['POST'])
def add_user_route() -> tuple[str, int]:
    """
    Endpoint to add a new user to the database.

    Returns:
        tuple[str, int]: Result message indicating success
            or failure of the operation and the status code.
    """
    try:
        username = request.form['username']
        password = request.form['password']

        # Validate input
        if not username or not password:
            return 'Username and password are required', 400

        # Escape user input to prevent XSS
        username = escape(username)
        password = escape(password)

        result = add_user(username, password)

        # Check if the addition was successful
        if result:
            return 'User added successfully', 200
        else:
            return 'Failed to add user', 500

    except KeyError:
        return 'Invalid input', 400
    except Exception as e:
        # Log the exception for internal tracking
        # And do not expose the exception message to the user
        app.logger.error(f'Error adding user: {e}')
        return 'An internal error occurred', 500


@app.route('/delete_user/<username>', methods=['DELETE'])
def delete_user_route(username: str) -> tuple[str, int]:
    """
    Endpoint to delete an existing user from the database.

    Args:
        username (str): Username of the user to be deleted.

    Returns:
        tuple[str, int]: Result message indicating success
            or failure of the operation and the status code.
    """
    try:
        # Validate input
        if not username:
            return 'Username is required', 400

        # Escape user input to prevent XSS
        username = escape(username)

        result = delete_user(username)

        # Check if the deletion was successful
        if result:
            return 'User deleted successfully', 200
        else:
            return 'Failed to delete user', 500

    except Exception as e:
        # Log the exception for internal tracking
        # And do not expose the exception message to the user
        app.logger.error(f'Error deleting user {username}: {e}')
        return 'An internal error occurred', 500


@app.route('/update_username', methods=['PUT'])
def update_username_route() -> tuple[str, int]:
    """
    Endpoint to update a user's username in the database.

    Returns:
        tuple[str, int]: Result message indicating success
            or failure of the operation and the status code.
    """
    try:
        old_username = request.form['old_username']
        new_username = request.form['new_username']

        # Validate input
        if not old_username or not new_username:
            return 'Old username and new username are required', 400

        # Escape user input to prevent XSS
        old_username = escape(old_username)
        new_username = escape(new_username)

        result = update_username(old_username, new_username)

        # Check if the update was successful
        if result:
            return 'Username updated successfully', 200
        else:
            return 'Failed to update username', 500

    except KeyError:
        return 'Invalid input', 400
    except Exception as e:
        # Log the exception for internal tracking
        # And do not expose the exception message to the user
        app.logger.error(f'Error updating username: {e}')
        return 'An internal error occurred', 500


@app.route('/update_password', methods=['PUT'])
def update_password_route() -> tuple[str, int]:
    """
    Endpoint to update a user's password in the database.

    Returns:
        tuple[str, int]: Result message indicating success
            or failure of the operation and the status code.
    """
    try:
        username = request.form['username']
        new_password = request.form['new_password']

        # Validate input
        if not username or not new_password:
            return 'Username and new password are required', 400

        # Escape user input to prevent XSS
        username = escape(username)
        new_password = escape(new_password)

        result = update_password(username, new_password)

        # Check if the update was successful
        if result:
            return 'Password updated successfully', 200
        else:
            return 'Failed to update password', 500

    except KeyError:
        return 'Invalid input', 400
    except Exception as e:
        # Log the exception for internal tracking
        # And do not expose the exception message to the user
        app.logger.error(f'Error updating password: {e}')
        return 'An internal error occurred', 500


if __name__ == '__main__':
    # Ensure the database tables are created at startup
    with app.app_context():
        db.create_all()
    # Run the application on all available IPs at port 6000
    app.run(host='0.0.0.0', port=6000)
