import { JwksClient } from "jwks-rsa";
import jwt, { Secret } from "jsonwebtoken";
import {doc,getDoc, getDocs, getFirestore, collection, query, setDoc, where, updateDoc, serverTimestamp, Firestore} from 'firebase/firestore'

const appId = process.env.CANVA_APP_ID || ""
const CACHE_EXPIRY_MS = 60 * 60 * 1_000; // 60 minutes
const TIMEOUT_MS = 30 * 1_000; // 30 seconds

export interface DesignTokenInterface{
  designId:string;
}

export interface UserTokenInterface{
  userId:string;
  brandId:string;
}

export async function addDesignsToDB(db:Firestore,access_token:string,continual?:string|undefined){
  //NOTE this is to be refreshed with a fresh access token every time
  const collectionRef = collection(db,"designs")
  const url = continual ? `https://api.canva.com/rest/v1/designs?continuation=${continual}` : `https://api.canva.com/rest/v1/designs`
  const re = await fetch(url, { method: "GET", headers: { "Authorization": `Bearer ${access_token}`} })
  const {} = await re.json()
  const designs = await re.json()
  const {continuation,items} = designs
  for(const item of items){
    const {id,owner,thumbnail,title,urls} = item
    const {url,width,height} = thumbnail
    const {user_id,team_id} = owner
    const {edit_url,view_url} = urls
    await setDoc(doc(db, "designs", id), {
      title,
      user_id,
      team_id,
      edit_url,
      view_url,
      thumbnail_width:width,
      thumbnail_height:height,
      thumbnail_url:url,
      timestamp: serverTimestamp()
    })
  }
  if(continuation){
    return await addDesignsToDB(db,access_token,continuation)
  }
}

export function tokenize(){
    const token = jwt.sign({
      iss: process.env.FIREBASE_ADMIN_CLIENT_EMAIL || "",
      sub: process.env.FIREBASE_ADMIN_CLIENT_EMAIL || "",
      aud: "https://identitytoolkit.googleapis.com/google.identity.identitytoolkit.v1.IdentityToolkit",
      iat: Date.now()/1000,
      exp: Date.now()/1000 + 3600,
      uid: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || ""
    },process.env.FIREBASE_ADMIN_PRIVATE_KEY||"",{algorithm:"RS256"})
    return token
}

export async function verifyUserToken(token: string) {
    const publicKey = await getPublicKey({ appId, token })
    if(!publicKey) return
    return jwt.verify(token, publicKey, {
      audience: appId,
      ignoreNotBefore: true
    });
  }
  
export async function verifyDesignToken(token: string) {
    const publicKey = await getPublicKey({ appId, token })
    if(!publicKey) return
    return jwt.verify(token, publicKey, {
      audience: appId,
      ignoreNotBefore: true
    });
}
  
export async function getPublicKey({ appId, token, cacheExpiryMs = CACHE_EXPIRY_MS, timeoutMs = TIMEOUT_MS}: {
    appId: string;
    token: string;
    cacheExpiryMs?: number;
    timeoutMs?: number;
  }):Promise<Secret | undefined> {
    const decoded = jwt.decode(token, {
      complete: true,
    });

    if(!decoded) return
  
    const jwks = new JwksClient({
      cache: true,
      cacheMaxAge: cacheExpiryMs,
      timeout: timeoutMs,
      rateLimit: true,
      jwksUri: `https://api.canva.com/rest/v1/apps/${appId}/jwks`,
    });
  
    const key = await jwks.getSigningKey(decoded.header.kid);
    return key.getPublicKey();
  }
  
export function getTokenFromHeader(request: Request) {
    const header = request.headers.get("authorization");
  
    if (!header) {
      return
    }
    const parts = header.split(" ");
  
    if (parts.length !== 2 || parts[0].toLowerCase() !== "bearer") {
      return;
    }
    const [, token] = parts
    return token;
}
