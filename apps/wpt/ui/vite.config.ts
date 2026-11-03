import { defineConfig } from "vite";

// No plugin: Vite turns TSX into JavaScript itself, following tsconfig.json's "jsx".
export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
});
