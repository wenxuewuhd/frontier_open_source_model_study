import { defineConfig } from 'astro/config';

// GitHub Pages 项目站点，所以带 base。dev 与 build 用同一个 base，
// 避免本地能跑、部署后链接全错这种分叉。
export default defineConfig({
  site: 'https://wenxuewuhd.github.io',
  base: process.env.SITE_BASE ?? '/frontier_open_source_model_study',
  outDir: './dist',
  build: { format: 'directory' },
});
