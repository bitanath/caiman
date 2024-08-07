import * as Functions from '@/app/api/utils' 
import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {DesignTokenInterface,UserTokenInterface} from "../../utils"
import {doc,getDoc,getDocs,collection,query,where,getFirestore, updateDoc, setDoc} from 'firebase/firestore'
import {app} from '@/../firebase'
import { tokenize } from '../../utils'

export const dynamic = 'force-dynamic'
export async function POST(request: Request,{ params }: { params: { designToken: string } }) {
  try{
    const header = request.headers.get("authorization")
    const userToken = header?.split(" ")[1]
    const {designToken} = params
    if(!userToken || !designToken) return Response.json({ "error":"Unable to retrieve Design" },{status:403})
    const verifiedDesignToken = await Functions.verifyDesignToken(designToken)
    const verifiedUserToken = await Functions.verifyUserToken(userToken)
    if(!verifiedDesignToken) return Response.json({ "error":"Unable to retrieve Design" },{status:403})
    if(!verifiedUserToken) return Response.json({ "error":"Unable to retrieve User" },{status:403})
    const {designId} = verifiedDesignToken as DesignTokenInterface
    const {brandId,userId} = verifiedUserToken as UserTokenInterface

    await signInWithCustomToken(getAuth(app),tokenize())
    const db = getFirestore(app)
    const collectionRef = collection(db,"tokens")
    let queried = query(collectionRef)
    queried = query(queried,where("canva_user","==",userId))
    const querySnapshot = await getDocs(queried)
    if(querySnapshot.empty) return Response.json({ "error":"User not linked" },{status:403})
    const {access_token,refresh_token,uid,user_id,team_id} = querySnapshot.docs[0].data()
    if(!access_token || !refresh_token) return Response.json({ "error":"User not connected" },{status:403})

    
    const searchParams = new URLSearchParams()
    searchParams.append('grant_type','refresh_token')
    searchParams.append('refresh_token',refresh_token)
    searchParams.append('scope','design:content:read design:meta:read')

    const client_id=process.env.CANVA_CONNECT_ID || ""
    const client_secret=process.env.CANVA_CONNECT_SECRET || ""
    const basicHeader = "Basic "+Buffer.from(`${client_id}:${client_secret}`).toString("base64")

    const resp = await fetch("https://api.canva.com/rest/v1/oauth/token", {
      method: "POST",
      headers: {
          "Authorization": basicHeader,
          "Content-Type": "application/x-www-form-urlencoded",
      },
      body:searchParams
    })
    const data = await resp.json();
    if(!data.access_token || !data.refresh_token) return Response.json({ "error":"User not connected" },{status:403})
    console.log("Scopes:",data.scope)
    await updateDoc(querySnapshot.docs[0].ref,{access_token:data.access_token,refresh_token:data.refresh_token,expires_in:data.expires_in})

    console.log("Got new access token ",data.access_token)

    const fetcher = await fetch("https://api.canva.com/rest/v1/exports", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "design_id": `${designId}`,
        "format": {
          "type": "jpg",
          "quality": 90
        }
      }),
    })
    const {job} = await fetcher.json()
    

    if(!job || !job.id || !job.status) return Response.json({ "error":"Job is not available" },{status:403})
    
    const urls = await monitorJob(access_token,job.id)
    return Response.json({ "urls":urls },{status:200})

  }catch(e:any){
    return Response.json({ "error":e.toString() },{status:403})
  }
}


function monitorJob(access_token:string,job_id:string){
  return new Promise((resolve) => {
    const interval = setInterval(async () => {
      
      const response = await fetch(`https://api.canva.com/rest/v1/exports/${job_id}`, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${access_token}`,
        },
      })
      
      const json = await response.json()
      const {job} = json
      const {id,status} = job
      
      if(status == "success"){
        const urls = job.urls
        clearInterval(interval)
        return resolve(urls);
      }
    }, 500);
  });
}