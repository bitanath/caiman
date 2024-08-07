import * as Functions from '@/app/api/utils' 
import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {DesignTokenInterface,UserTokenInterface} from "../../utils"
import {doc,getDoc,getDocs,collection,query,where,getFirestore, updateDoc, setDoc} from 'firebase/firestore'
import {app} from '@/../firebase'
import { tokenize } from '../../utils'

export const dynamic = 'force-dynamic'
export async function POST(request: Request,{ searchParams }: { searchParams: { designToken: string } }) {
  try{
    const header = request.headers.get("authorization")
    const userToken = header?.split(" ")[1]
    const {designToken} = searchParams
    const body = await request.json();

    console.log("Got this",designToken,userToken,body)

    if(!userToken || !designToken) return Response.json({ "error":"Unable to retrieve Design or User" },{status:403})

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
    let queried = query(collectionRef)
    queried = query(queried,where("app_user_id","==",userId))
    const querySnapshot = await getDocs(queried)
    const user = querySnapshot.docs[0]

    const uid = user.get("uid")
    const refresh_token = user.get("refresh_token")

    if(!uid || !refresh_token){
      return Response.json({ "error": "Unable to find user linked to design"},{status:403})
    }

    //NOW to request this design using tokens
    const params = new URLSearchParams()
    params.append('grant_type','refresh_token')
    params.append('refresh_token',refresh_token)

    const client_id=process.env.CANVA_CONNECT_ID || ""
    const client_secret=process.env.CANVA_CONNECT_SECRET || ""
    const basicHeader = "Basic "+Buffer.from(`${client_id}:${client_secret}`).toString("base64")

    const resp = await fetch("https://api.canva.com/rest/v1/oauth/token", {
      method: "POST",
      headers: {
          "Authorization": basicHeader,
          "Content-Type": "application/x-www-form-urlencoded",
      },
      body:params
    })
    const data = await resp.json();
    const {access_token,scope,expires_in} = data
    await updateDoc(user.ref,{access_token,refresh_token:data.refresh_token,expires_in})

    return Response.json({ designId,brandId,userId},{status:200})
  }catch(e:any){
    return Response.json({ "error": e.toString()},{status:403})
  }
}
