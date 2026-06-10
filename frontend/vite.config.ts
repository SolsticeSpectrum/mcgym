import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev proxy to the python monitor so the app can use relative paths
const monitor = process.env.MONITOR || 'http://localhost:9080'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: Object.fromEntries(
      ['/state', '/agent', '/palette'].map((p) => [p, { target: monitor, changeOrigin: true }]),
    ),
  },
})
