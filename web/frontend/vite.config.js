import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Frontend dev server proxies /api to the FastAPI backend on :8000,
// so the React app can call /api/scan without CORS hassle in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
