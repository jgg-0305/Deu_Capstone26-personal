/*
 * 프런트엔드의 개발 서버와 빌드 환경을 설정한다. 개발 중 브라우저 API 요청을 로컬 FastAPI로 중계한다.
 */
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
