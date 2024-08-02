"use client"
import { HStack,VStack,Spacer,Box,Text } from "@kuma-ui/core"
import { GradientHeading } from "@/components/lib/gradient-heading"
import { BackgroundBeams } from "@/components/lib/background-beams";
import { ResetComponent } from "@/components/ui/reset";
import Navbar from "@/components/ui/navbar";

export default function Reset() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center pt-24">
      <Navbar></Navbar>
      <VStack justifyContent={"center"} alignItems={"center"}>
          <Box className="z-10">
            <ResetComponent></ResetComponent>
          </Box>
        <BackgroundBeams></BackgroundBeams>
      </VStack>
    </main>
  );
}

