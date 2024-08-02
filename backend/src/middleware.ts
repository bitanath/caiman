import { NextRequest, NextResponse } from "next/server";
import { authMiddleware, redirectToHome, redirectToLogin } from "next-firebase-auth-edge";
import { clientConfig, serverConfig } from "./config";

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
    handleValidToken: async ({token, decodedToken}, headers) => {
      if (AUTH_PATHS.includes(request.nextUrl.pathname)) {
        return NextResponse.redirect(new URL('/dashboard', request.url))
      }else{
        return NextResponse.next({
          request: {
            headers
          }
        })
      }
    },
    handleInvalidToken: async (reason) => {
      if (PUBLIC_PATHS.includes(request.nextUrl.pathname)) {
        return NextResponse.next();
      }else{
        return redirectToLogin(request, {
          path: '/login',
          publicPaths: PUBLIC_PATHS
        })
      }
    },
    handleError: async (error) => {
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
    "/api/logout",
  ],
};