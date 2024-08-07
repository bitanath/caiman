import * as crypto from 'crypto'

import { app } from "@/../firebase";
import { clientConfig, serverConfig } from "@/config";
import { getAuth, signInWithCustomToken } from "firebase/auth";

import { getTokens } from "next-firebase-auth-edge";
import { cookies,headers } from "next/headers";
import { notFound } from "next/navigation";

import { HStack, Spacer } from "@kuma-ui/core";
import { Button } from "@/components/ui/button";

import { Separator } from "@/components/ui/separator";
import AccountInfo from './accountinfo';
import Integrations from './integrations';
import { CreditCardIcon } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, } from "@/components/ui/card";

export default async function Auth() {
  const tokens = await getTokens(cookies(), {
    apiKey: clientConfig.apiKey,
    cookieName: serverConfig.cookieName,
    cookieSignatureKeys: serverConfig.cookieSignatureKeys,
    serviceAccount: serverConfig.serviceAccount,
  });
  if (!tokens) notFound();
  const auth = getAuth(app);
  const response = await signInWithCustomToken(auth, tokens?.customToken);
  const displayName = response.user.displayName || "John Doe";
  const email = response.user.email || "john@doe.com";
  const emailVerified = response.user.emailVerified || false;
  const uid = response.user.uid;

  const cookie = cookies()
  const canvaUser = cookie.get("x-canva")?.value

  return (
    <div className="flex min-h-screen w-screen">
      <div className="flex flex-col sm:gap-4">
        <main className="grid flex-1 items-start gap-4 md:gap-8 pt-8 pb-8 px-4">
          <AccountInfo displayName={displayName} email={email} customToken={tokens!.customToken}></AccountInfo>
          <Card className="col-span-2 h-[300px]">
            <CardHeader>
              <CardTitle>Subscription Details</CardTitle>
              <CardDescription>
                Caiman is free for the duration of the Hackathon.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Pro Plan</div>
                    <div className="text-sm text-muted-foreground">
                      Billed monthly
                    </div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">$0</div>
                    <div className="text-sm text-muted-foreground">
                      per month
                    </div>
                  </div>
                </div>
                <Separator />
                <div className="grid gap-2">
                  <div className="flex items-center justify-between">
                    <div>Next Billing </div>
                    <div><HStack gap={4}> <CreditCardIcon></CreditCardIcon> <span>Free</span></HStack></div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button variant="outline">Update Subscription</Button>
            </CardFooter>
          </Card>
          <Integrations emailVerified={emailVerified} uid={uid} customToken={tokens!.customToken} canvaUser={canvaUser}></Integrations>
        </main>
      </div>
    </div>
  );
}

