import type { Metadata } from "next";
import { ThemeProvider } from "@/components/themeprovider"
import { Inter } from "next/font/google";
import Navbar from "@/components/ui/navbar";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Canary for Canva",
  description: "Get Realtime interactive feedback on your design",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange >
        {children}
      </ThemeProvider>
      </body>
    </html>
  );
}
