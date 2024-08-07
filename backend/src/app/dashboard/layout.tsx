import { PageWithSidebar } from "@/components/ui/sidebarpage";
import { HStack,VStack,Spacer } from "@kuma-ui/core";

import { getTokens } from "next-firebase-auth-edge";
import { getAuth,signInWithCustomToken,signOut } from "firebase/auth";
import { notFound } from "next/navigation";
import { app } from "@/../firebase";
import { cookies } from "next/headers";
import { clientConfig, serverConfig } from "@/config";
import { Toaster } from "@/components/ui/toaster";

export default async function DashboardLayout({
    children
  }: {
    children: React.ReactNode
  }) {
    const tokens = await getTokens(cookies(), {
        apiKey: clientConfig.apiKey,
        cookieName: serverConfig.cookieName,
        cookieSignatureKeys: serverConfig.cookieSignatureKeys,
        serviceAccount: serverConfig.serviceAccount,
    });
    if(!tokens) notFound()
    const auth = getAuth(app);
    const response = await signInWithCustomToken(auth,tokens?.customToken);
    const displayName = response.user.displayName || "John Doe"

    return (
        <PageWithSidebar name={displayName}>
            <VStack>
                {children}
            </VStack>
            <Toaster richColors closeButton position="top-center"></Toaster>
        </PageWithSidebar>
    )
  }