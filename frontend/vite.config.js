import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The monitor (Python) serves /state, /agent, /palette with CORS enabled. In dev we proxy them so
// the app can use same-origin relative paths; override the target with MCAI_MONITOR.
const monitor = process.env.MCAI_MONITOR || 'http://localhost:9080'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: Object.fromEntries(
      ['/state', '/agent', '/palette'].map((p) => [p, { target: monitor, changeOrigin: true }]),
    ),
  },
})
