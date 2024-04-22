from .models import db, User

def add_user(username, password):
    # 创建一个新的用户实例
    new_user = User(username=username)
    new_user.set_password(password)  # 设置密码
    db.session.add(new_user)
    try:
        db.session.commit()  # 提交到数据库
        return f"User {username} added successfully."
    except Exception as e:
        db.session.rollback()  # 如果有错误发生，则回滚
        return f"Error adding user: {str(e)}"

def delete_user(username):
    user = User.query.filter_by(username=username).first()  # 根据用户名查找用户
    if user:
        db.session.delete(user)
        try:
            db.session.commit()  # 提交更改
            return f"User {username} deleted successfully."
        except Exception as e:
            db.session.rollback()  # 如果有错误发生，则回滚
            return f"Error deleting user: {str(e)}"
    else:
        return f"User {username} not found."

def update_username(old_username, new_username):
    user = User.query.filter_by(username=old_username).first()  # 根据旧用户名查找用户
    if user:
        user.username = new_username  # 更新用户名
        try:
            db.session.commit()  # 提交更改
            return f"Username updated successfully to {new_username}."
        except Exception as e:
            db.session.rollback()  # 如果有错误发生，则回滚
            return f"Error updating username: {str(e)}"
    else:
        return f"User {old_username} not found."

def update_password(username, new_password):
    user = User.query.filter_by(username=username).first()  # 根据用户名查找用户
    if user:
        user.set_password(new_password)  # 设置新密码
        try:
            db.session.commit()  # 提交更改
            return f"Password updated successfully for user {username}."
        except Exception as e:
            db.session.rollback()  # 如果有错误发生，则回滚
            return f"Error updating password: {str(e)}"
    else:
        return f"User {username} not found."
