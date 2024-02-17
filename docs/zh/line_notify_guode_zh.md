# 如何使用 LINE Notify

本指南提供了一個簡單的說明，闡述如何使用 LINE Notify 從各種應用程式發送訊息到您的 LINE 帳戶。LINE Notify 允許您在配置後直接在 LINE 上接收來自網絡服務和應用程式的通知。在開始之前，請確保您已生成個人 LINE Notify 令牌。

## 前提條件

- LINE 帳戶
- 個人 LINE Notify 令牌（本指南中稱為 `YOUR_LINE_NOTIFY_TOKEN`）

## 第 1 步：生成 LINE Notify 令牌

1. 訪問 [LINE Notify 網站](https://notify-bot.line.me/zh_TW/)。
2. 使用您的 LINE 憑證登入。
3. 導航至「我的頁面」區塊。
4. 點擊「生成令牌」按鈕。
5. 選擇所需服務並為令牌提供一個名稱。
6. 點擊「生成」。請記得安全地保存您的令牌。

## 第 2 步：安裝所需工具

在本教程中，我們將使用 `curl` 來展示如何發送通知。請確保您的系統上已安裝 `curl`。大多數 Unix 類操作系統，包括 Linux 和 macOS，都預裝了 `curl`。

## 第 3 步：使用 LINE Notify 發送訊息

要向您的 LINE 發送通知，請在您的終端機使用以下 `curl` 命令。將 `YOUR_LINE_NOTIFY_TOKEN` 替換為您在第 1 步中生成的令牌。

```bash
curl -X POST https://notify-api.line.me/api/notify \
     -H 'Authorization: Bearer YOUR_LINE_NOTIFY_TOKEN' \
     -F 'message=你好！這是來自 LINE Notify 的測試訊息。'
```

## 第 4 步：使用 Python 自動化通知

要自動化發送 LINE Notify 通知，您可以使用提供的 Python 範例程式碼 `src/line_notifier.py`。此腳本展示了如何使用 LINE Notify API 程式化發送訊息。

### 執行 Python 腳本的前提條件：

- 系統上安裝了 Python。
- 安裝了 `requests` 庫。您可以使用 pip 安裝：`pip install requests`。

### 如何使用 Python 腳本：

1. 確保您將腳本中的 `'YOUR_LINE_NOTIFY_TOKEN'` 替換為您實際的 LINE Notify 令牌。
2. 自訂您希望發送的訊息。範例腳本發送一條帶有時間戳的警告訊息。
3. 通過在您的終端機運行 `python src/line_notifier.py` 執行腳本。

### 腳本解釋：

- 腳本定義了一個函數 `send_line_notification`，該函數接受 LINE Notify 令牌和一條訊息作為參數。
- 它向 LINE Notify API 發送一個帶有指定訊息的 POST 請求。
- 該函數從 API 返回 HTTP 狀態碼，指示通知傳遞的成功或失敗。

```python
# send_line_notification 函數的範例使用：
line_token = 'YOUR_LINE_NOTIFY_TOKEN'  # 替換為您實際的 LINE Notify 令牌
message = '你好，這是一條測試訊

息。'
status = send_line_notification(line_token, message)
print(status)  # 打印 HTTP 狀態碼（例如，200 表示成功）
```

## 第 5 步：驗證收據

成功執行後，檢查您的 LINE 應用。您應該會收到來自腳本或 `curl` 命令發送的訊息。

## 故障排除

- **令牌無效**：確保您的令牌輸入正確。如果持續失敗，請重新生成令牌並再試一次。
- **未收到訊息**：檢查您的網路連接，並確保您的網路未限制 LINE。

欲了解更多詳細信息和高級用法，請參閱 [官方 LINE Notify 文件](https://notify-bot.line.me/doc/en/)。