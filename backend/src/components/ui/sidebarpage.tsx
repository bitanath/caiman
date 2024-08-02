"use client";
import React, { useState,useEffect } from "react";
import { Sidebar, SidebarBody, SidebarLink, SidebarButton } from "@/components/lib/sidebar";
import { useTheme } from "next-themes";

import { useRouter } from "next/navigation";
import { getAuth, signOut } from "firebase/auth";
import { app } from "../../../firebase";

import { ArrowLeft as IconArrowLeft, User2Icon as IconUserBolt, MessageSquareDotIcon as  IconBrandTabler, SunIcon, MoonIcon} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

import Link from "next/link";

export function PageWithSidebar({
    children
}:{
    children: React.ReactNode
}) {
  const router = useRouter();
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
      <SidebarButton label={<span>Theme</span>} icon={theme == "dark" ? <SunIcon className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" /> : <MoonIcon className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" />} onClick={()=>setTheme(theme == "dark" ? "light" : "dark")}></SidebarButton>
    )
  }

  async function handleLogout() {
    await signOut(getAuth(app));

    await fetch("/api/logout");

    router.push("/login");
  }


  const links = [
    {
      label: "Account",
      href: "#",
      icon: ( <IconUserBolt className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" /> ),
      type: "link"
    },
    {
      label: "History",
      href: "#",
      icon: ( <IconBrandTabler className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" /> ),
      type: "link"
    },
    {
      label: "Logout",
      href: "#",
      icon: ( <IconArrowLeft className="text-neutral-700 dark:text-neutral-200 h-5 w-5 flex-shrink-0" /> ),
      type: "button"
    },
  ];
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "flex flex-col md:flex-row bg-gray-100 dark:bg-neutral-800 w-full h-full flex-1 overflow-hidden",
        "h-screen w-screen" 
      )}
    >
      <Sidebar open={open} setOpen={setOpen}>
        <SidebarBody className="justify-between gap-10">
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
            {open ? <Logo name={"Bitan Nath"}/> : <LogoIcon />}
            <div className="mt-8 flex flex-col gap-2">
              {links.map((link, idx) => (
                link.type == "link" ? <SidebarLink key={idx} link={link} /> : <SidebarButton key={idx} label={link.label} icon={link.icon} onClick={handleLogout}></SidebarButton>
              ))}
            </div>
          </div>
          <div className="">
            <ThemeButton></ThemeButton>
          </div>
        </SidebarBody>
      </Sidebar>
      <Dashboard>
        {children}
      </Dashboard>
    </div>
  );
}

export const Logo = ({name}:{
  name:string
}) => {
  return (
    <Link
      href="#"
      className="font-normal flex space-x-2 items-center text-sm text-black py-1 relative z-20"
    >
      <div className="h-5 w-6 bg-black dark:bg-white rounded-lg rounded-sm rounded-lg rounded-sm flex-shrink-0 text-xs text-center items-center pt-[2px]">BN</div>
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="font-medium text-black dark:text-white whitespace-pre"
      >
        {name}
      </motion.span>
    </Link>
  );
};

export const LogoIcon = () => {
  return (
    <Link
      href="#"
      className="font-normal flex space-x-2 items-center text-sm text-black py-1 relative z-20"
    >
      <div className="h-5 w-6 bg-black dark:bg-white rounded-lg rounded-sm rounded-lg rounded-sm flex-shrink-0 text-xs text-center items-center pt-[2px]">BN</div>
    </Link>
  );
};


const Dashboard = ({
    children
}:{
    children: React.ReactNode
}) => {
  return (
    <div className="flex flex-1">
      <div className="p-2 md:p-10 rounded-tl-2xl border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 flex flex-col gap-2 flex-1 w-full h-full">
        {children}
      </div>
    </div>
  );
};
