🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 用戶管理系統

這個用戶管理系統是一個基於Flask的網絡應用程式，提供了管理用戶帳戶的端點。它允許在資料庫中使用SQLAlchemy ORM添加、刪除和更新用戶憑證。

## 使用方法

要啟動應用程式，請導航至包含`app.py`的目錄，並執行以下命令：

```sh
flask run --host=0.0.0.0 --port=6000
```

這將在所有可用的IP上啟動Flask開發伺服器，端口為6000。

### 端點

應用程式提供以下端點：

- `POST /add_user`：在資料庫中添加一個新用戶。需要表單數據，包括`username`和`password`。

- `DELETE /delete_user/<username>`：根據提供的用戶名從資料庫中刪除現有用戶。

- `PUT /update_username`：更新用戶的用戶名。需要表單數據，包括`old_username`和`new_username`。

- `PUT /update_password`：更新用戶的密碼。需要表單數據，包括`username`和`new_password`。

### 示例使用方法

要添加用戶，您可以使用如下的`curl`命令：

```sh
curl -X POST -d "username=johndoe&password=securepassword" http://localhost:6000/add_user
```

要刪除用戶：

```sh
curl -X DELETE http://localhost:6000/delete_user/johndoe
```

更新用戶名：

```sh
curl -X PUT -d "old_username=johndoe&new_username=johnsmith" http://localhost:6000/update_username
```

更新密碼：

```sh
curl -X PUT -d "username=johnsmith&new_password=newsecurepassword" http://localhost:6000/update_password
```

請注意，這些命令僅供說明之用，應根據您的具體情況和安全要求進行調整。

## 配置

在運行應用程式之前，您必須配置資料庫連接。應用程式使用Flask-SQLAlchemy進行資料庫交互。

要配置資料庫URI，請在`app.py`中更新`SQLALCHEMY_DATABASE_URI`：

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'your_database_uri_here'
```

將`'your_database_uri_here'`替換為您資料庫的實際URI。例如，如果您在開發中使用SQLite，它可能看起來像這樣：

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database_file.db'
```

確保將`SQLALCHEMY_TRACK_MODIFICATIONS`設置為`False`，以抑制不必要的警告並禁用消耗額外記憶體的SQLAlchemy事件系統。

## 附加資訊

用戶模型定義在`models.py`中，包括安全設置和驗證密碼的方法。密碼使用Werkzeug的安全助手進行哈希處理，以確保它們不以純文本形式存儲。

`user_management.py`文件包含與用戶模型交互的邏輯，以及執行添加、刪除和更新用戶等操作。

如需進一步的幫助或查詢，請參閱Flask和SQLAlchemy的文檔。
