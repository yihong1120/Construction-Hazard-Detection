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
    // Specify the output directory for the build process
    outDir: '../dist',
  },
});
