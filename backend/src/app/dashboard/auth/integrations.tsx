"use client"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, } from "@/components/ui/card";
import { HStack, Spacer } from "@kuma-ui/core";
import { Button } from "@/components/ui/button";
import { Separator } from "@radix-ui/react-separator";
import { ChevronRightIcon } from "lucide-react";
import { IconProps } from "@/components/ui/toolbar";
import { usePathname } from "next/navigation";
import { toast } from "sonner";

import * as crypto from 'crypto'
import base64url from 'base64url'

import { signInWithCustomToken,getAuth, sendEmailVerification } from 'firebase/auth'
import {doc,getDocs,setDoc,collection,getFirestore,query,where, serverTimestamp, updateDoc} from 'firebase/firestore'
import {app} from '@/../firebase'
import { timeStamp } from "console";
import { useEffect,useState } from "react";

export default function Integrations({
    emailVerified,
    uid,
    customToken,
    canvaUser
}:{
    emailVerified:boolean;
    uid:string;
    customToken: string;
    canvaUser?: string|null;
}){
    const [linkedUser,setLinkedUser] = useState(null)

    useEffect(()=>{
        (async () => {
          //TODO: Update Canva User Id for user
          await signInWithCustomToken(getAuth(app),customToken)
          const db = getFirestore(app)
          const collectionRef = collection(db,"tokens")
          let queried = query(collectionRef,where("uid","==",uid))
          const querySnapshot = await getDocs(queried)
          if(querySnapshot.empty) return
          const docRef = querySnapshot.docs[0].ref
          const userId = querySnapshot.docs[0].get("user_id")
          const canvaUserId = querySnapshot.docs[0].get("canva_user")
          //A combination of canva user and user id can tell you if user is linked
          if(canvaUser){
            updateDoc(docRef,{canva_user:canvaUser})
          }else if(canvaUserId){
            canvaUser = canvaUserId
          }
          if(userId){
            setLinkedUser(userId)
          }
        })();
        return () => {
          //cleanups here for unmounting
        }
    })

    const handleConnect = async ()=>{
      if(linkedUser && canvaUser){
        toast.loading("Reconnecting to Canva")
        await connectToCanva()
      }else if(linkedUser){
        toast.warning("Connected to Canva but not linked to Design. Open from the Caiman app in Canva to continue.")
      }else{
        toast.loading("Connecting to Canva")
        await connectToCanva()
      }
    }

    const connectToCanva = async ()=>{
        //XXX Change this before deployment
        let redirect_uri = window.location.hostname == "localhost" ? "https://127.0.0.1:3001/api/oauth" : `${window.location.hostname}/api/oauth`
        redirect_uri = encodeURIComponent(redirect_uri)

        const madeid = makeid(47)
        const code_verifier = base64url.encode(madeid)
        //store the id in a table
        let credential = await signInWithCustomToken(getAuth(app),customToken)
        const db = getFirestore(app)
        const collectionRef = collection(db,"tokens")
        let queried = query(collectionRef)
        queried = query(queried,where("uid","==",uid))
        const querySnapshot = await getDocs(queried)
        console.log("Got query docs",querySnapshot.docs.length,"uid",madeid)
        if(querySnapshot.empty){
            const createdDoc = await setDoc(doc(collectionRef),{
                uid,
                code_verifier:madeid,
                timeStamp: serverTimestamp()
            })
        }else{
          updateDoc(querySnapshot.docs[0].ref,{
            code_verifier:madeid,
            timeStamp: serverTimestamp()
          })
        }

        console.log("Using redirect uri ",redirect_uri,uid,code_verifier)
        const state = encodeURIComponent(uid);
        const codeChallenge = base64url.encode(crypto.createHash("sha256").update(code_verifier).digest())
        
        const url = `https://www.canva.com/api/oauth/authorize?code_challenge_method=s256&response_type=code&client_id=OC-AZDd_7zghFQI&redirect_uri=${redirect_uri}&scope=design:meta:read%20design:content:read%20design:permission:read%20profile:read&code_challenge=${codeChallenge}&state=${state}`
        console.log("Using url ",url)
        const win = window.open(url, '_blank');
        if (win == null || win.closed || typeof win.closed=='undefined') {
          toast.error("You might have popups disabled. Please enable to continue.")
        }else{
          win.focus();
        }
    }

    const displayConnection = async ()=>{
      if(canvaUser){
        return toast.success("Connected! Please use the Caiman App in Canva to Continue")
      }
      let credential = await signInWithCustomToken(getAuth(app),customToken)
      await handleConnect()
    }

    const resendVerification = async ()=>{
      try{
        if(emailVerified) return

        let credential = await signInWithCustomToken(getAuth(app),customToken)
        if(credential.user.emailVerified){
          return toast.success("Email Verified Successfully")
        }
      
        await sendEmailVerification(credential.user)
        toast.info("Resent Verification Email")
      }catch(e:any){
        toast.error(e.toString())
      }
      
    }

    return (
        <Card className="col-span-2 h-[300px]">
        <CardHeader>
          <CardTitle>Integrations</CardTitle>
          <CardDescription>
            Connect your account to third-party services.
          </CardDescription>
          <Separator />
          <CardDescription className="text-xs italic">
            Note: Email verification and Canva integration are mandatory.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CanvaIcon className="h-5 w-5 text-muted-foreground" />
                <div>Canva Integration</div>
              </div>
              <div className="text-xs text-muted-foreground px-2">{linkedUser && canvaUser ? "connected" : canvaUser ? "linked, but not connected" : linkedUser ? "connected, but not linked" : "disconnected"}</div>
              <Button variant="ghost" size="icon">
                <ChevronRightIcon className="h-5 w-5" onClick={displayConnection}/>
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <EmailIcon className="h-5 w-5 text-muted-foreground" />
                <div>Email Verification</div>
              </div>
              <div className="text-xs text-muted-foreground">{emailVerified ? "verified" : "mail sent"}</div>
              <Button variant="ghost" size="icon" onClick={resendVerification}>
                <ChevronRightIcon className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={handleConnect}>{linkedUser && canvaUser ? "Reconnect to Canva" : "Connect to Canva"}</Button>
        </CardFooter>
      </Card>
    )
}

function CanvaIcon(props: IconProps) {
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
  
  function EmailIcon(props: IconProps) {
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
        <rect width="3" height="8" x="13" y="2" rx="1.5" />
        <path d="M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5" />
        <rect width="3" height="8" x="8" y="14" rx="1.5" />
        <path d="M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5" />
        <rect width="8" height="3" x="14" y="13" rx="1.5" />
        <path d="M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5" />
        <rect width="8" height="3" x="2" y="8" rx="1.5" />
        <path d="M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5" />
      </svg>
    );
  }
  

  function makeid(length:number) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    let counter = 0;
    while (counter < length) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength));
      counter += 1;
    }
    return result;
}