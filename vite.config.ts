import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

// `workbench.html` is a localdev-only agent instrument driven by Playwright MCP
// (see CLAUDE.md §Workbench conventions). Dev still serves it — any .html in
// the project root is reachable during `vite` / `vite dev` — but the production
// build only ships the human-facing demo at `/`.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/gemma4-webgpu/' : '/',
  server: {
    port: 5175,
    strictPort: true,
    host: '127.0.0.1',
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
      },
    },
  },
}));
