import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js Proxy — handles CSP headers and API auth.
 *
 * 1. CSP: sets per-request security headers.
 *    Dev mode:  allows 'unsafe-inline' + 'unsafe-eval' (Turbopack HMR).
 *    Prod mode: 'unsafe-inline' only (RSC inline scripts need it).
 *
 * 2. API Auth: injects VOCALIE_API_KEY as x-api-key header on /v1/*
 *    requests so the browser never sees the key.
 */

function generateNonce(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(18));
  return btoa(String.fromCharCode(...bytes));
}

export function proxy(request: NextRequest) {
  const nonce = generateNonce();
  const isDev = process.env.NODE_ENV === "development";
  const apiKey = process.env.VOCALIE_API_KEY || "";

  // ── API requests: inject auth header ──
  if (request.nextUrl.pathname.startsWith("/v1/")) {
    const requestHeaders = new Headers(request.headers);
    if (apiKey) {
      requestHeaders.set("x-api-key", apiKey);
    }
    return NextResponse.next({
      request: { headers: requestHeaders },
    });
  }

  // ── Page requests: CSP + security headers ──
  const scriptSrc = isDev
    ? "'self' 'unsafe-inline' 'unsafe-eval'"
    : "'self' 'unsafe-inline'";

  const csp = [
    "default-src 'self'",
    "base-uri 'self'",
    "frame-ancestors 'none'",
    "form-action 'self'",
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "font-src 'self' data:",
    `script-src ${scriptSrc}`,
    "style-src 'self' 'unsafe-inline'",
    "connect-src 'self' http://127.0.0.1:8018 http://localhost:8018 ws://127.0.0.1:8018 ws://localhost:8018",
  ].join("; ");

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);

  const response = NextResponse.next({
    request: { headers: requestHeaders },
  });

  response.headers.set("Content-Security-Policy", csp);
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  response.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=()"
  );

  if (!isDev) {
    response.headers.set(
      "Strict-Transport-Security",
      "max-age=63072000; includeSubDomains; preload"
    );
  }

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};