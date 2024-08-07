"use client"

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

import { app } from "@/../firebase";
import { getAuth, createUserWithEmailAndPassword, updateProfile, sendEmailVerification } from "firebase/auth";

import { HStack,VStack,Spacer,Box,Text } from "@kuma-ui/core"
import { BackgroundBeams } from "@/components/lib/background-beams";
import { SignupComponent } from "@/components/ui/signup";
import Navbar from "@/components/ui/navbar";
import { Toaster } from "@/components/ui/toaster"
import { toast } from "sonner"
 

export default function Signup() {
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    let reject = false

    if (password !== confirmation) {
      setError("Passwords don't match");
      toast.error("Passwords don't match")
      reject = true
    }

    if(firstName == ""){
      setError("First Name is required");
      toast.error("First Name is required")
      reject = true
    }
    if(lastName == ""){
      setError("Last Name is required");
      toast.error("Last Name is required")
      reject = true
    }
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

    console.log(error)

    if(reject){
      return
    }

    try {
      toast.loading("Signing up")
      let response = await createUserWithEmailAndPassword(getAuth(app), email, password)
      const user = response.user
      await updateProfile(user,{displayName: firstName+ " " + lastName})
      console.log("Updated profile with display name",user,firstName+ " " + lastName)
      await sendEmailVerification(user)
      toast.info("Signed Up Successfully")
      router.push("/login");
    } catch (e) {
      setError((e as Error).message);
      toast.error((e as Error).message)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center pt-24">
      <Navbar></Navbar>
      <VStack justifyContent={"center"} alignItems={"center"}>
          <Box className="z-10">
            <SignupComponent email={email} setEmail={setEmail} password={password} setPassword={setPassword} confirmation={confirmation} setConfirmation={setConfirmation} firstname={firstName} setFirstname={setFirstName} lastname={lastName} setLastname={setLastName} submit={handleSubmit}></SignupComponent>
          </Box>
        <BackgroundBeams></BackgroundBeams>
      </VStack>
      <Toaster richColors closeButton position="top-center"></Toaster>
    </main>
  );
}

