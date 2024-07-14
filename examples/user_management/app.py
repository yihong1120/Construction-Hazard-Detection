from __future__ import annotations

from flask import Flask
from flask import request
from user_operation import add_user
from user_operation import delete_user
from user_operation import update_password
from user_operation import update_username

from models import db

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
def add_user_route() -> str:
    """
    Endpoint to add a new user to the database.

    Returns:
        str: Result message indicating success or failure of the operation.
    """
    username = request.form['username']
    password = request.form['password']
    result = add_user(username, password)
    return result


@app.route('/delete_user/<username>', methods=['DELETE'])
def delete_user_route(username: str) -> str:
    """
    Endpoint to delete an existing user from the database.

    Args:
        username (str): Username of the user to be deleted.

    Returns:
        str: Result message indicating success or failure of the operation.
    """
    result = delete_user(username)
    return result


@app.route('/update_username', methods=['PUT'])
def update_username_route() -> str:
    """
    Endpoint to update a user's username in the database.

    Returns:
        str: Result message indicating success or failure of the operation.
    """
    old_username = request.form['old_username']
    new_username = request.form['new_username']
    result = update_username(old_username, new_username)
    return result


@app.route('/update_password', methods=['PUT'])
def update_password_route() -> str:
    """
    Endpoint to update a user's password in the database.

    Returns:
        str: Result message indicating success or failure of the operation.
    """
    username = request.form['username']
    new_password = request.form['new_password']
    result = update_password(username, new_password)
    return result


if __name__ == '__main__':
    # Ensure the database tables are created at startup
    with app.app_context():
        db.create_all()
    # Run the application on all available IPs at port 6000
    app.run(host='0.0.0.0', port=6000)
