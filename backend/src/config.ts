export const serverConfig = {
    cookieName: process.env.AUTH_COOKIE_NAME|| "",
    cookieSignatureKeys: [process.env.AUTH_COOKIE_SIGNATURE_KEY_CURRENT|| "secret_caiman1", process.env.AUTH_COOKIE_SIGNATURE_KEY_PREVIOUS|| "secret_caiman2"],
    cookieSerializeOptions: {
      path: "/",
      httpOnly: true,
      secure: process.env.USE_SECURE_COOKIES === "true",
      sameSite: "lax" as const,
      maxAge: 12 * 60 * 60 * 24,
    },
    serviceAccount: {
      projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "",
      clientEmail: process.env.FIREBASE_ADMIN_CLIENT_EMAIL|| "",
      privateKey: process.env.FIREBASE_ADMIN_PRIVATE_KEY|| "",
    }
};
  
export const clientConfig = {
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID|| "",
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY|| "",
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN|| "",
    databaseURL: process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL|| "",
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID|| ""
};

export const adminConfig = {
    type: "service_account",
    project_id: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "",
    private_key_id: process.env.FIREBASE_AUTH_TOKEN_ID || "",
    private_key: process.env.FIREBASE_ADMIN_PRIVATE_KEY || "",
    client_email: process.env.FIREBASE_ADMIN_CLIENT_EMAIL || "",
    client_id: process.env.FIREBASE_CLIENT_ID || "",
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token",
    auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
    client_x509_cert_url: process.env.NEXT_PUBLIC_FIREBASE_CLIENT_X509 || "",
    universe_domain: "googleapis.com"
}