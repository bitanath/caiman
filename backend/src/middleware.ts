"use server"
import { NextRequest, NextResponse } from "next/server";
import { authMiddleware, redirectToHome, redirectToLogin } from "next-firebase-auth-edge";
import { clientConfig, serverConfig } from "./config";

import {app} from '@/../firebase'
import { signInWithCustomToken,getAuth,updateProfile } from 'firebase/auth'
import {doc,getDocs,setDoc,collection,getFirestore,query,where, serverTimestamp, updateDoc} from 'firebase/firestore'
import { cookies } from "next/headers";

const PUBLIC_PATHS = ['/', '/signup', '/login', '/reset'];
const AUTH_PATHS = ['/signup', '/login', '/reset'];

export default function middleware(request: NextRequest) {
  return authMiddleware(request, {
    loginPath: "/api/login",
    logoutPath: "/api/logout",
    apiKey: clientConfig.apiKey,
    cookieName: serverConfig.cookieName,
    cookieSignatureKeys: serverConfig.cookieSignatureKeys,
    cookieSerializeOptions: serverConfig.cookieSerializeOptions,
    serviceAccount: serverConfig.serviceAccount,
    handleValidToken: async ({token, decodedToken, customToken}, headers) => {
      let canvaUser = request.nextUrl.searchParams.get("canva")
      // console.log("MIDDLEWARE:",canvaUser,cookies().get("x-canva"))
      
      if (AUTH_PATHS.includes(request.nextUrl.pathname)) {
        const response = NextResponse.redirect(new URL(`/dashboard`, request.url))
        if(canvaUser){
          response.cookies.set("x-canva",canvaUser)
        }
        return response
      }else{
        const response = NextResponse.next({
          request: {
            headers
          }
        })
        if(canvaUser){
          response.cookies.set("x-canva",canvaUser)
        }
        return response
      }
    },
    handleInvalidToken: async (reason) => {
      let canvaUser = request.nextUrl.searchParams.get("canva")
      // console.log("MIDDLEWARE NO AUTH:",canvaUser,cookies().get("x-canva"))
      
      if (PUBLIC_PATHS.includes(request.nextUrl.pathname)) {
        const response = NextResponse.next()
        if(canvaUser){
          response.cookies.set("x-canva",canvaUser)
        }
        return response
      }else{
        return redirectToLogin(request, {
          path: canvaUser ? '/login' : `/login?canva=${canvaUser}`,
          publicPaths: PUBLIC_PATHS
        })
      }
    },
    handleError: async (error) => {
      console.log("MIDDLEWARE ERROR AUTH:")
      console.log('Unhandled authentication error', {error});
      
      return redirectToLogin(request, {
        path: '/login',
        publicPaths: PUBLIC_PATHS
      });
    }
  });
}

export const config = {
  matcher: [
    "/",
    "/dashboard",
    "/dashboard/:path",
    "/((?!_next|api|.*\\.).*)",
    "/api/login",
    "/api/logout"
  ],
};