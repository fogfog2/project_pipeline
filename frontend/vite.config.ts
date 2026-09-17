import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_STATIC_MODE === "true" ? (process.env.VITE_BASE_PATH || "/") : "/",
  server: { port: 5173, proxy: { "/api": `http://127.0.0.1:${process.env.VITE_API_PORT || "8000"}` } },
  build: { outDir: "dist", sourcemap: true }
});
