import Link from "next/link";

import { HeroCaption } from "@/components/ui/hero"
import { AppToolbar } from "@/components/ui/toolbar";
import { HStack,VStack,Spacer,Box,Text } from "@kuma-ui/core"
import { GradientHeading } from "@/components/lib/gradient-heading"
import { AnimatedGradientButton } from "@/components/lib/animated-gradient-text"
import { BackgroundBeams } from "@/components/lib/background-beams";
import { ShimmeringCTA } from "@/components/ui/shimmer-cta";
import Navbar from "@/components/ui/navbar";

export default function Home() {

  return (
    <main className="flex min-h-screen flex-col items-center justify-center pt-24">
      <Navbar></Navbar>
      <VStack justifyContent={"center"} alignItems={"center"}>
          <Box className="z-10">
            <ShimmeringCTA>
              <Link href="https://github.com/bitanath/canary">
                <span>✨ Built for Canva AI and Integrations Hackathon</span>
              </Link>
            </ShimmeringCTA>
            <GradientHeading variant="default" size="xxxl" weight="bold" className="text-center">
              AI Design Critic
            </GradientHeading>
            <Spacer height={4}></Spacer>
            <HeroCaption></HeroCaption>
            <VStack>
              <Link href="/dashboard">
                <AnimatedGradientButton emoji="🔐">Login to Connect with Canva</AnimatedGradientButton>
              </Link>
            </VStack>
          </Box>
        <BackgroundBeams></BackgroundBeams>
      </VStack>
    </main>
  );
}

