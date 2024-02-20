🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# LINE 聊天機器人範例

此儲存庫提供創建使用 Flask 和 LINE Messaging API 的 LINE 聊天機器人的範例。此範例展示如何設置一個基本的聊天機器人，它會回應收到的相同文字訊息。

## 使用方法

### 設定環境變數

在啟動您的聊天機器人之前，您需要設定必要的環境變數：

1. `LINE_CHANNEL_ACCESS_TOKEN`：您的 LINE 頻道存取令牌。
2. `LINE_CHANNEL_SECRET`：您的 LINE 頻道密鑰。

這些可以在創建您的 LINE 機器人後從 LINE 開發者控制台獲得。按照以下步驟來檢索您的憑證：

#### LINE 開發者控制台設置

1. 訪問 [LINE 開發者控制台](https://developers.line.biz/console/)。
2. 使用您的 LINE 帳戶登錄。
3. 如果您尚未創建，則創建一個新的提供者和機器人。
4. 從列表中選擇您的機器人頻道。
5. 在頻道設置中，找到頻道密鑰和頻道存取令牌部分。
6. 複製頻道密鑰並生成頻道存取令牌（如果您尚未生成）。

您可以在環境中設定這些變數，或者在 `line_bot.py` 腳本中替換 `'YOUR_LINE_CHANNEL_ACCESS_TOKEN'` 和 `'YOUR_LINE_CHANNEL_SECRET'` 為您實際的 LINE 頻道存取令牌和密鑰。

### 運行聊天機器人

要運行聊天機器人，首先確保您已安裝 Flask，或使用 pip 安裝：

```bash
pip install Flask
```

然後，運行 `line_bot.py` 腳本：

```bash
python line_bot.py
```

Flask 應用將啟動，您的聊天機器人將在本地運行。

### Webhook 設定

為了讓 LINE 平台與您的聊天機器人溝通，您需要設置一個 webhook URL。這個 URL 應指向您的 Flask 應用正在運行的 `/callback` 端點。

1. 返回 LINE 開發者控制台並選擇您的機器人。
2. 在設定中，找到 Webhook 設定部分。
3. 將 Webhook URL 設置為您的 Flask 應用運行的 URL，後面跟上 `/callback`。例如，`https://yourappname.ngrok.io/callback`。
4. 驗證 webhook。

如果您在本地測試，可以使用如 [ngrok](https://ngrok.com/) 這樣的服務將您的本地服務器暴露到互聯網上。

設置您的 webhook 後，任何發送給您 LINE 機器人的訊息都將被回應。

## 功能

- **Flask 網頁應用**：使用 Flask 創建可以處理來自 LINE 的請求的網頁服務器。
- **LINE Messaging API**：利用 LINE Messaging API 接收和回應訊息。
- **Webhook 處理**：包含如何處理來自 LINE 的 webhook 事件的範例。
- **訊息回音**：聊天機器人回應任何收到的

文字訊息。

## 配置

您可以通過修改 `line_bot.py` 中的 `handle_message` 函數來自定義聊天機器人的回應或添加更複雜的功能。

請記得保護您的 webhook 端點，特別是如果您將聊天機器人部署到生產環境。

## 部署

對於部署，您可以使用任何支持 Python 和 Flask 的雲平台，如 Heroku、AWS 或 Google Cloud。記得在 LINE 開發者控制台中使用您部署的應用 URL 更新您的 webhook URL。

確保您的部署環境中正確設定了環境變數。

## 進一步開發

這個範例提供了一個基本設置。對於更進階的功能，如豐富訊息、快速回覆和範本訊息，請參考 [LINE Messaging API 文件](https://developers.line.biz/en/docs/messaging-api/)。

探索添加功能，如用戶認證、資料庫整合和更複雜的對話流程，以增強您的聊天機器人。