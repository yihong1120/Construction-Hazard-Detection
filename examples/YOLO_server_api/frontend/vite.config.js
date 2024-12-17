import { defineConfig } from 'vite'

export default defineConfig({
  root: 'public', // 指定靜態檔案目錄
  server: {
    host: '127.0.0.1',
    port: 7777,
    // fs: {
    //   strict: true // 嚴格文件系統模式，避免自動解析 `.html`
    // },
    proxy: {
      // 僅針對 API 的代理，不影響靜態文件的訪問
      '/api': {
        target: 'http://changdar-server.mooo.com:28000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: 'public/index.html',
        login: 'public/login.html',
        detection: 'public/detection.html',
        model_management: 'public/model_management.html',
        user_management: 'public/user_management.html',
        header: 'public/header.html',
        footer: 'public/footer.html'
      }
    }
  }
})
