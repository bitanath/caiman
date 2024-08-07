"use client"
import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

import { app } from "@/../firebase";
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";

import { HStack,VStack,Spacer,Box,Text } from "@kuma-ui/core"
import { BackgroundBeams } from "@/components/lib/background-beams"
import { LoginComponent } from "@/components/ui/login"
import Navbar from "@/components/ui/navbar"

import { Toaster } from "@/components/ui/toaster"
import { toast } from "sonner"

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");

    let reject = false

    if(email == ""){
      setError("Email is required");
      toast.error("Email is required")
      reject = true
    }
    if(password == ""){
      setError("Password is required");
      toast.error("Password is required")
      reject = true
    }

    if(reject) return
    toast.promise(signInWithEmailAndPassword(
      getAuth(app),
      email,
      password
    ), {
      loading: 'Logging In...',
      success: (credential) => {
        credential.user.getIdToken().then(async idToken=>{
          if(!idToken) return toast.error("Errored out while logging in!")
          await fetch("/api/login", {
            headers: {
              Authorization: `Bearer ${idToken}`,
            },
          })
          router.replace("/dashboard")
        })
        return "Logged In... Now Redirecting"
      },
      error: 'Errored out while logging in',
    });
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center pt-24">
      <Navbar></Navbar>
      <VStack justifyContent={"center"} alignItems={"center"}>
          <Box className="z-10">
            <LoginComponent email={email} setEmail={setEmail} password={password} setPassword={setPassword} submit={handleSubmit}></LoginComponent>
          </Box>
        <BackgroundBeams></BackgroundBeams>
      </VStack>
      <Toaster richColors closeButton position="top-center"></Toaster>
    </main>
  );
}

