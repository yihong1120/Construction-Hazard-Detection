
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 串流網頁前端範例

這個專案展示了一個用於串流攝影機畫面的網頁前端，使用 HTML、CSS、JavaScript 和 Vite 建構。前端與後端 API 互動，以獲取並顯示攝影機串流和相關的元數據。

## 使用方法

要開始使用這個專案，請按照以下步驟操作：

### 1. 安裝 Node.js 和 npm

確保您已安裝 Node.js 和 npm。您可以從 [Node.js 官方網站](https://nodejs.org/) 下載並安裝。建議使用 LTS（長期支援）版本。

#### 在 Windows 上安裝

1. **下載 Windows 安裝程式**：
    - 前往 [Node.js 官方網站](https://nodejs.org/)。
    - 點擊 LTS 版本以下載安裝程式。

2. **執行安裝程式**：
    - 打開下載的 `.msi` 檔案。
    - 按照安裝程式中的提示操作。接受許可協議，選擇安裝路徑，並確保勾選將 Node.js 添加到 PATH 的選項。

3. **驗證安裝**：
    - 打開命令提示字元或 PowerShell。
    - 執行以下命令以檢查 Node.js 和 npm 是否安裝成功：
      ```sh
      node -v
      npm -v
      ```
    - 這些命令應該會顯示您系統上安裝的 Node.js 和 npm 的版本。

4. **更新 npm（可選但建議）**：
    - 有時候，Node.js 附帶的 npm 版本可能會過時。您可以通過以下命令更新 npm 到最新版本：
      ```sh
      npm install -g npm
      ```

#### 在 macOS 上安裝

1. **下載 macOS 安裝程式**：
    - 前往 [Node.js 官方網站](https://nodejs.org/)。
    - 點擊 LTS 版本以下載安裝程式。

2. **執行安裝程式**：
    - 打開下載的 `.pkg` 檔案。
    - 按照安裝程式中的提示操作。

3. **或者，使用 Homebrew**：
    - 如果您偏好使用套件管理器，您可以通過 Homebrew 安裝 Node.js：
      ```sh
      brew install node
      ```

4. **驗證安裝**：
    - 打開終端機。
    - 執行以下命令以檢查 Node.js 和 npm 是否安裝成功：
      ```sh
      node -v
      npm -v
      ```

#### 在 Linux 上安裝

對於基於 Debian 的發行版（例如 Ubuntu），您可以使用以下命令：

1. **添加 NodeSource 倉庫**：
    ```sh
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    ```

2. **安裝 Node.js 和 npm**：
    ```sh
    sudo apt-get install -y nodejs
    ```

3. **驗證安裝**：
    - 打開終端機。
    - 執行以下命令以檢查 Node.js 和 npm 是否安裝成功：
      ```sh
      node -v
      npm -v
      ```

對於其他發行版，請參考 [Node.js 安裝指南](https://nodejs.org/en/download/package-manager/)。

### 2. 克隆倉庫

使用 Git 將倉庫克隆到您的本地機器：
```sh
git clone https://your-gitea-server.com/username/repository.git
cd repository/examples/streaming_web/frontend
```

### 3. 安裝依賴

導航到專案目錄並運行以下命令以安裝所需的依賴：
```sh
npm install
```

### 4. 啟動開發伺服器

運行以下命令以在端口 8888 上啟動 Vite 開發伺服器：
```sh
npm run dev
```
您應該會看到伺服器正在運行的輸出。打開瀏覽器並導航到 `http://localhost:8888` 以查看應用程式。

### 5. 構建生產版本

運行以下命令以構建專案的生產版本：
```sh
npm run build
```
構建的檔案將輸出到 `dist` 目錄。然後，您可以使用靜態檔案伺服器提供這些檔案或將它們部署到網頁伺服器。

## 功能

- **響應式設計**：前端設計為響應式，適應各種螢幕尺寸，確保在桌面和移動設備上都有良好的用戶體驗。
- **動態內容**：攝影機標籤和串流動態從後端 API 獲取並顯示在前端。
- **WebSocket 集成**：通過 WebSocket 連接處理實時更新，確保攝影機串流和警告實時更新。
- **錯誤處理**：前端包括錯誤處理，以管理如缺少 URL 參數和 WebSocket 連接錯誤等問題。
- **無障礙設計**：設計考慮了無障礙性，例如圖像的替代文字和高對比度模式調整。

## 配置

專案配置通過 `vite.config.js` 文件管理。主要配置選項包括：

- **根目錄**：專案的根目錄設置為 `public`。
- **開發伺服器**：開發伺服器配置為在端口 8888 上運行，並設置代理以將 API 請求轉發到運行在 `http://127.0.0.1:8000` 的後端伺服器。
- **構建輸出**：構建輸出目錄設置為 `dist`，位於 `public` 目錄之外。Rollup 選項指定了主、標籤和攝影機 HTML 文件的入口點。

### `vite.config.js` 範例

```javascript
import { defineConfig } from 'vite';

// Export the Vite configuration object
export default defineConfig({
  // Set the root directory for the project to 'public'
  root: 'public',

  server: {
    // Define the port for the development server
    port: 8888,

    proxy: {
      // Configure proxy settings for the '/api' prefix
      '/api': {
        // The backend server to forward API requests to
        target: 'http://127.0.0.1:8000',

        // Enable changing the origin of the host header to the target URL
        changeOrigin: true,

        // Enable WebSocket proxying
        ws: true,
      },
    },
  },

  build: {
    outDir: '../dist', // Output directory outside of public
    rollupOptions: {
      input: {
        main: './public/index.html', // Main entry
        label: './public/label.html', // Additional entry
        camera: './public/camera.html', // Additional entry
      },
    },
  },
});
```

## 套件資訊

專案使用以下 `package.json` 配置：

```json
{
    "name": "streaming-web-frontend",
    "version": "1.0.0",
    "description": "Frontend for the streaming web application with FastAPI backend.",
    "main": "index.js",
    "scripts": {
        "dev": "vite",
        "build": "vite build",
        "preview": "vite preview"
    },
    "keywords": [],
    "author": "yihong1120",
    "license": "AGPL-3.0-only",
    "dependencies": {
        "axios": "^1.5.0"
    },
    "devDependencies": {
        "vite": "^5.4.11"
    }
}
```

### 腳本

- **`dev`**：啟動 Vite 開發伺服器。
- **`build`**：構建專案的生產版本。
- **`preview`**：預覽構建的專案。

### 依賴

- **`axios`**：一個基於 Promise 的 HTTP 客戶端，用於瀏覽器和 Node.js。

### 開發依賴

- **`vite`**：一個構建工具，旨在為現代網頁專案提供更快、更精簡的開發體驗。

## 檔案結構

專案組織如下：

- **public/**：包含前端的 HTML、CSS 和 JavaScript 檔案。
  - **index.html**：顯示攝影機標籤的主頁。
  - **label.html**：顯示特定標籤詳細信息的頁面。
  - **camera.html**：顯示特定攝影機實時串流的頁面。
  - **css/**：包含樣式表（`styles.css`）的目錄。
  - **js/**：包含 JavaScript 檔案（`index.js`、`label.js`、`camera.js`）的目錄。
- **package.json**：套件配置文件。
- **vite.config.js**：Vite 配置文件。

## 其他資訊

有關如何使用和擴展此專案的更多詳細信息，請參考源代碼文件中的註釋。註釋提供了對專案中實現的各個部分和功能的解釋。

### 開發提示

- **熱模塊替換（HMR）**：Vite 支持 HMR，這意味著您對代碼所做的更改將在瀏覽器中反映，而無需完全重新加載頁面。這加快了開發速度並提供了即時反饋。
- **調試**：使用瀏覽器開發者工具來調試 JavaScript 代碼。您可以設置斷點、檢查變量並逐步執行代碼以了解其工作原理。
- **代碼檢查和格式化**：考慮使用 ESLint 和 Prettier 等工具來保持代碼質量和一致性。這些工具可以集成到您的開發工作流程中，自動檢查和格式化您的代碼。

### 部署

要部署構建的檔案，您可以使用任何靜態檔案伺服器或網頁託管服務。以下是一些選項：

- **Netlify**：部署靜態網站的熱門選擇。您可以連接您的 Git 倉庫，Netlify 將自動構建和部署您的網站。
- **Vercel**：另一個流行的平台，用於部署靜態網站和無伺服器功能。它也與 Git 倉庫很好地集成。
- **GitHub Pages**：如果您的倉庫託管在 GitHub 上，您可以使用 GitHub Pages 直接從倉庫提供您的靜態網站。
