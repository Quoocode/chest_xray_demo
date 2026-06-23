import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";
import type { ReactNode } from "react";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
});

export const metadata: Metadata = {
  title: "Chest X-ray AI Assistant",
  description: "Demo web app for multi-label chest X-ray classification.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={spaceGrotesk.variable}>
      <body>{children}</body>
    </html>
  );
}
