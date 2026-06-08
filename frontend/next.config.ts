import type { NextConfig } from "next";

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8018";
const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: {
    root: __dirname,
  },
  // CSP & security headers are handled by src/middleware.ts (nonce-based).
  // Do NOT set CSP here — it would conflict with the middleware.
  async rewrites() {
    return [
      {
        source: "/v1/:path*",
        destination: `${apiBase}/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
