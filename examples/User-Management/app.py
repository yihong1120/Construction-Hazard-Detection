from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from user_management import add_user, delete_user, update_username, update_password
from .models import db

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'your_database_uri_here'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 在第一次运行应用前
with app.app_context():
    db.create_all()

@app.route('/add_user', methods=['POST'])
def add_user_route():
    username = request.form['username']
    password = request.form['password']
    result = add_user(username, password)
    return result

@app.route('/delete_user/<username>', methods=['DELETE'])
def delete_user_route(username):
    result = delete_user(username)
    return result

@app.route('/update_username', methods=['PUT'])
def update_username_route():
    old_username = request.form['old_username']
    new_username = request.form['new_username']
    result = update_username(old_username, new_username)
    return result

@app.route('/update_password', methods=['PUT'])
def update_password_route():
    username = request.form['username']
    new_password = request.form['new_password']
    result = update_password(username, new_password)
    return result

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=6000)
