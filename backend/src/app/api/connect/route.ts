import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {doc,getDoc,getFirestore} from 'firebase/firestore'
import {app} from '@/../firebase'
import jwt from 'jsonwebtoken'

export const dynamic = 'force-dynamic'
export async function GET(request: Request) {
  const response = await signInWithCustomToken(getAuth(app),tokenize())
  const idToken = await response.user.getIdToken()
  let db = getFirestore(app)
  const docRef = doc(db, "designs", "89FijI7MM9H60JGcKQgI");
  const docSnap = await getDoc(docRef);
  return Response.json({ "teri": process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID, "maka":docSnap.data() })
}

function tokenize(){
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