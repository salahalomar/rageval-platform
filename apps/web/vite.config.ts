import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

// 5173 is Vite's default and the right answer on a clean machine, but developers
// routinely have another Vite project already holding it -- in which case this one binds
// the wildcard address, loses the race to localhost and serves someone else's app with no
// error anywhere. WEB_PORT is the escape hatch.
const port = Number(process.env.WEB_PORT ?? 5173);

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // reachable from outside the container
    port,
    strictPort: true, // fail loudly rather than silently serving from a neighbour's port
    // The /eval page imports the committed ablation JSON straight out of eval/results
    // rather than keeping a copy, so the dev server has to be allowed to read above its
    // own root. The build has no such restriction.
    fs: { allow: [repoRoot] },
    // The API is proxied rather than called cross-origin so the browser never needs
    // CORS and the deployed build can sit behind one host. VITE_API_TARGET lets the
    // compose service point at the `api` service name instead of localhost.
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  preview: { port, strictPort: true },
});
