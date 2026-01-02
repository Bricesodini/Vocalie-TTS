This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## API configuration

The frontend uses a simple precedence rule for API access:

1) If `NEXT_PUBLIC_API_BASE` is set, requests go directly to that absolute base URL.
2) Otherwise, the app uses relative `/v1/*` paths which are proxied by Next.js
   via the rewrite in `next.config.ts`.

For local dev, leave `NEXT_PUBLIC_API_BASE` unset and keep the API running on
`http://127.0.0.1:8000` so the rewrite works out of the box.

## NPM-only workspace

This frontend is npm-only. If you see lockfile/SWC warnings, run a clean npm reinstall:

```
rm -rf node_modules .next
npm ci
```

Yarn is not required.

## Lockfile warning

If Next.js warns about multiple lockfiles, remove the accidental global
`~/package-lock.json` so Turbopack uses the local `frontend/` lockfile.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
