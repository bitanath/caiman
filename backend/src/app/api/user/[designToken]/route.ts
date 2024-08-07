import * as Functions from '@/app/api/utils' 
import {DesignTokenInterface,UserTokenInterface,tokenize} from "../../utils"

import { signInWithCustomToken,getAuth, } from 'firebase/auth'
import {getDoc, getDocs, getFirestore, collection, query, where, updateDoc, serverTimestamp} from 'firebase/firestore'
import {app} from '@/../firebase'


export const dynamic = 'force-dynamic'
export async function GET(request: Request,{ params }: { params: { designToken: string } }) {
  try{
    const header = request.headers.get("authorization")
    const userToken = header?.split(" ")[1]
    const {designToken} = params

    if(!userToken || !designToken) return Response.json({ "error": "Unable to find user or design"},{status:403})

    const verifiedDesignToken = await Functions.verifyDesignToken(designToken)
    const verifiedUserToken = await Functions.verifyUserToken(userToken)
    if(!verifiedDesignToken) return Response.json({ "error":"Unable to retrieve Design" },{status:403})
    if(!verifiedUserToken) return Response.json({ "error":"Unable to retrieve User" },{status:403})

    const {designId} = verifiedDesignToken as DesignTokenInterface
    const {brandId,userId} = verifiedUserToken as UserTokenInterface
    if(!designId) return Response.json({ "error":"Unable to retrieve Design Id" },{status:403})
    if(!userId) return Response.json({ "error":"Unable to retrieve Design Id" },{status:403})

    await signInWithCustomToken(getAuth(app),tokenize())
    const db = getFirestore(app)
    const collectionRef = collection(db,"tokens")
    let queried = query(collectionRef,where("canva_user","==",userId))
    const snapshot = await getDocs(queried)
    if(snapshot.empty){
      return Response.json({ "error":"Design not Linked",designId,brandId,userId },{status:207})
    }
    const uuId = snapshot.docs[0].get("uid")

    return Response.json({ designId,brandId,userId,uuId},{status:200})
  }catch(e:any){
    return Response.json({ "error": e.toString()},{status:403})
  }
}
