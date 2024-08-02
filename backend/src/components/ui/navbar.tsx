"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";



export default function Navbar() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  function ThemeButton() {
    if(!mounted){
      return null 
    }
    return (
      <Button
        variant="ghost"
        size="icon"
        className="rounded-full hover:bg-gray-800/50 dark:hover:bg-gray-700/50"
        onClick={() => setTheme(theme == "dark" ? "light" : "dark")}
      >
        {theme == "dark" ? <SunIcon /> : <MoonIcon />}
        <span className="sr-only">Toggle theme</span>
      </Button>
    );
  }

  return (
    <header className="fixed top-0 left-0 z-50 w-full flex items-center justify-between bg-slate-300/20 px-4 py-3 text-gray-50 shadow-sm dark:bg-gray-900/20 sm:px-6 lg:px-8">
      <div className="flex items-center gap-4">
        <Link href="/" className="flex items-center gap-2" prefetch={false}>
          <VidbanditIcon className="h-6 w-6" />
          <span className="text-lg font-semibold text-gray-500 dark:text-white">
            Canary for Canva
          </span>
        </Link>
        <a
          href="https://github.com/bitanath/giva"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center rounded-md border border-gray-800 bg-gray-800 px-3 py-1 text-sm font-medium transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-950 focus:ring-offset-2 dark:border-gray-700 dark:bg-gray-700 dark:hover:bg-gray-600 dark:focus:ring-gray-300"
        >
          <GithubIcon className="mr-2 h-4 w-4" />
          Fork
        </a>
      </div>
      <div className="flex items-center gap-4">
          <ThemeButton></ThemeButton>
      </div>
    </header>
  );
}

export function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

export function MoonIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="stroke-gray-500 h-5 w-5"
    >
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

export function VidbanditIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      // stroke="currentColor"
      strokeWidth="1"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="stroke-slate-500 dark:stroke-white mb-1"
    >
      {/* <path d="m8 3 4 8 5-5 5 15H2L8 3z" /> */}
      <path d="M 3.175781 3.175781 C 3.074219 3.25 3 3.824219 3 4.398438 L 3 5.476562 L 2.074219 5.425781 L 1.125 5.375 L 1.226562 8.625 C 1.25 10.398438 1.324219 11.898438 1.351562 11.925781 C 1.375 11.949219 1.949219 11.699219 2.625 11.351562 C 4.074219 10.625 6.625 10.550781 6.625 11.25 C 6.625 11.523438 6.199219 11.648438 4.925781 11.773438 C 3.023438 11.925781 2.199219 12.476562 1.75 13.898438 C 1.199219 15.574219 2.800781 19.125 4.601562 20.226562 C 5.976562 21.074219 7.550781 20.976562 9.824219 19.898438 C 10.851562 19.398438 11.824219 19 12 19 C 12.175781 19 13.148438 19.398438 14.175781 19.898438 C 16.449219 20.976562 18.023438 21.074219 19.398438 20.226562 C 21.199219 19.125 22.800781 15.574219 22.25 13.898438 C 21.800781 12.476562 20.976562 11.925781 19.074219 11.773438 C 17.800781 11.648438 17.375 11.523438 17.375 11.25 C 17.375 10.550781 19.925781 10.625 21.375 11.351562 C 22.050781 11.699219 22.625 11.949219 22.648438 11.925781 C 22.675781 11.898438 22.726562 10.398438 22.773438 8.625 L 22.875 5.375 L 21.949219 5.398438 L 21 5.425781 L 21 4.324219 C 21 3.550781 20.875 3.199219 20.601562 3.074219 C 19.601562 2.699219 16.199219 5.5 14.449219 8.125 L 13.425781 9.699219 L 12.726562 9.273438 C 12.050781 8.898438 11.949219 8.898438 11.273438 9.300781 L 10.550781 9.699219 L 9.648438 8.375 C 8.375 6.425781 7.300781 5.300781 5.648438 4.074219 C 4.273438 3.050781 3.550781 2.800781 3.175781 3.175781 Z M 6.023438 6.300781 C 6.625 6.898438 7.398438 7.75 7.75 8.226562 C 8.273438 9 8.300781 9.074219 7.851562 8.898438 C 7.601562 8.800781 6.75 8.648438 6 8.574219 C 4.949219 8.5 4.398438 8.574219 3.699219 8.949219 C 3.023438 9.300781 2.75 9.351562 2.75 9.125 C 2.75 8.949219 2.675781 8.449219 2.574219 8 C 2.425781 7.199219 2.449219 7.175781 3.101562 7.351562 C 4.023438 7.574219 4.5 7.148438 4.5 6.101562 C 4.5 5.625 4.601562 5.25 4.699219 5.25 C 4.824219 5.25 5.398438 5.726562 6.023438 6.300781 Z M 19.550781 6.300781 L 19.625 7.375 L 20.601562 7.324219 C 21.476562 7.273438 21.550781 7.300781 21.398438 7.824219 C 21.324219 8.125 21.25 8.625 21.25 8.898438 L 21.25 9.449219 L 20.324219 8.949219 C 19.601562 8.574219 19.050781 8.5 18 8.574219 C 17.25 8.648438 16.425781 8.773438 16.148438 8.898438 C 15.75 9.050781 15.726562 9.023438 15.976562 8.523438 C 16.351562 7.851562 18.925781 5.25 19.25 5.25 C 19.375 5.25 19.5 5.726562 19.550781 6.300781 Z M 12.375 11.523438 C 12.375 12.125 12.25 12.375 12 12.375 C 11.601562 12.375 11.375 11.449219 11.601562 10.824219 C 11.851562 10.175781 12.375 10.625 12.375 11.523438 Z M 8.824219 14.523438 C 9.226562 14.949219 9.476562 15.425781 9.425781 15.574219 C 9.300781 15.976562 5.023438 16.125 4.648438 15.75 C 4.476562 15.574219 4.550781 15.300781 4.898438 14.824219 C 6.023438 13.300781 7.601562 13.175781 8.824219 14.523438 Z M 18.824219 14.523438 C 19.226562 14.949219 19.476562 15.425781 19.425781 15.574219 C 19.300781 15.976562 15.023438 16.125 14.648438 15.75 C 14.476562 15.574219 14.550781 15.300781 14.898438 14.824219 C 16.023438 13.300781 17.601562 13.175781 18.824219 14.523438 Z M 18.824219 14.523438 " />
    </svg>
  );
}

export function SunIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="stroke-slate-500 w-5 h-5"
    >
      <circle cx="12" cy="12" r="4"></circle>
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}
