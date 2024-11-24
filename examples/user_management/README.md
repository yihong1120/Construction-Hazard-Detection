🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# User Management System

This User Management System is a Flask-based web application that provides endpoints for managing user accounts. It allows for the addition, deletion, and updating of user credentials within a database using SQLAlchemy ORM.

## Usage

To start the application, navigate to the directory containing `app.py` and execute the following command:

```sh
uvicorn examples.user_management.app:app --host 127.0.0.1 --port 6000
```

This will start the Flask development server on all available IPs at port 6000.

### Endpoints

The application provides the following endpoints:

- `POST /add_user`: Adds a new user to the database. Requires form data with `username` and `password`.

- `DELETE /delete_user/<username>`: Deletes an existing user from the database based on the provided username.

- `PUT /update_username`: Updates a user's username. Requires form data with `old_username` and `new_username`.

- `PUT /update_password`: Updates a user's password. Requires form data with `username` and `new_password`.

### Example Usage

To add a user, you might use a `curl` command like the following:

```sh
curl -X POST -d "username=johndoe&password=securepassword" http://localhost:6000/add_user
```

To delete a user:

```sh
curl -X DELETE http://localhost:6000/delete_user/johndoe
```

To update a username:

```sh
curl -X PUT -d "old_username=johndoe&new_username=johnsmith" http://localhost:6000/update_username
```

To update a password:

```sh
curl -X PUT -d "username=johnsmith&new_password=newsecurepassword" http://localhost:6000/update_password
```

Please note that these commands are for illustrative purposes and should be adapted to your specific context and security requirements.

## Configuration

Before running the application, you must configure the database connection. The application uses Flask-SQLAlchemy for database interactions.

To configure the database URI, update the `SQLALCHEMY_DATABASE_URI` in `app.py`:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'your_database_uri_here'
```

Replace `'your_database_uri_here'` with the actual URI of your database. For example, if you are using SQLite for development, it might look like this:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database_file.db'
```

Ensure that `SQLALCHEMY_TRACK_MODIFICATIONS` is set to `False` to suppress unnecessary warnings and to disable event system of SQLAlchemy which consumes additional memory.

## Additional Information

The User model is defined in `models.py` and includes methods for setting and verifying passwords securely. Passwords are hashed using Werkzeug's security helpers to ensure they are not stored in plain text.

The `user_management.py` file contains the logic for interacting with the User model and performing operations such as adding, deleting, and updating users.

For any further assistance or queries, please refer to the Flask and SQLAlchemy documentation.
