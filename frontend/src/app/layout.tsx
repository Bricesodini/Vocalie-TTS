import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vocalie-TTS",
  description: "Interface Vocalie-TTS API + frontend capability-driven",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
