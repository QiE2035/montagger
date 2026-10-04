import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

// The dev server proxies API calls to the backend on :8457; production
// builds land in dist/ and are served by the backend itself.
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8457",
      "/health": "http://127.0.0.1:8457",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
