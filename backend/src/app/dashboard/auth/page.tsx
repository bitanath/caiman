import { PageWithSidebar } from "@/components/ui/sidebarpage";
import { HStack,VStack,Spacer } from "@kuma-ui/core";

import { getTokens } from "next-firebase-auth-edge";
import { cookies } from "next/headers";
import { notFound } from "next/navigation";
import { clientConfig, serverConfig } from "@/config";



export default async function Dashboard(){
    const tokens = await getTokens(cookies(), {
        apiKey: clientConfig.apiKey,
        cookieName: serverConfig.cookieName,
        cookieSignatureKeys: serverConfig.cookieSignatureKeys,
        serviceAccount: serverConfig.serviceAccount,
    });
    
    if (!tokens) {
        notFound();
    }
    return (
        <PageWithSidebar>
            <VStack>
                
            </VStack>
        </PageWithSidebar>
    )
}