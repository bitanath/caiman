import { JwksClient } from "jwks-rsa";
import jwt, { Secret } from "jsonwebtoken";

const appId = process.env.CANVA_APP_ID || ""
const CACHE_EXPIRY_MS = 60 * 60 * 1_000; // 60 minutes
const TIMEOUT_MS = 30 * 1_000; // 30 seconds

export async function verifyUserToken(token: string) {
    const publicKey = await getPublicKey({ appId, token })
    if(!publicKey) return
    return jwt.verify(token, publicKey, {
      audience: appId,
    });
  }
  
export async function verifyDesignToken(token: string) {
    const publicKey = await getPublicKey({ appId, token })
    if(!publicKey) return
    return jwt.verify(token, publicKey, {
      audience: appId,
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