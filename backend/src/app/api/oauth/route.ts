import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {getDoc, getDocs, getFirestore, collection, query, where, updateDoc, serverTimestamp} from 'firebase/firestore'
import { addDesignsToDB, tokenize } from '../utils'
import {app} from '@/../firebase'

import { type NextRequest } from 'next/server'
import { redirect } from 'next/navigation'
import { isRedirectError } from 'next/dist/client/components/redirect'
import { URLSearchParams } from 'url'
import base64url from 'base64url'

export async function GET(request: NextRequest) {
    try{
        const params = request.nextUrl.searchParams
        let code = params.get("code")
        if (!code) return Response.json({ "error": "No code found" })
        let state = params.get("state")
        if (!state) return Response.json({ "error": "No state found" })

        const client_id=process.env.CANVA_CONNECT_ID || ""
        const client_secret=process.env.CANVA_CONNECT_SECRET || ""
        //Now to authenticate for an access token
        const basicHeader = "Basic "+Buffer.from(`${client_id}:${client_secret}`).toString("base64")
        console.log("Got code",code)
        const uid = state
        console.log("Got details",uid,request.nextUrl.href)
        const redirect_uri = request.nextUrl.hostname == "localhost" ? "https://127.0.0.1:3001/api/oauth" : request.nextUrl.href

        //TODO now get stored code verification from challenge
        const response = await signInWithCustomToken(getAuth(app),tokenize())
        const db = getFirestore(app)
        const collectionRef = collection(db,"tokens")
        let queried = query(collectionRef)
        queried = query(queried,where("uid","==",uid))
        const querySnapshot = await getDocs(queried)
        const doc = querySnapshot.docs[0]
        const code_verifier = base64url.encode(doc.get("code_verifier"))

        console.log("Got code verifier ",code_verifier)
        
        const body = new URLSearchParams()
        body.append('grant_type','authorization_code')
        body.append('code_verifier',code_verifier)
        body.append('code',code)
        body.append('redirect_uri',redirect_uri)

        console.log("Authorization", basicHeader,body)
        const resp = await fetch("https://api.canva.com/rest/v1/oauth/token", {
            method: "POST",
            headers: {
                "Authorization": basicHeader,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body
        })
        const data = await resp.json();
        console.log("Got data ",data)
        const {access_token,refresh_token,expires_in,scope} = data
        //TODO now update the doc with the query string
        await updateDoc(doc.ref,{access_token,refresh_token,expires_in,scope,timeStamp: serverTimestamp()})

        const followup = await fetch("https://api.canva.com/rest/v1/users/me", {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${access_token}`,
                "Content-Type": "application/x-www-form-urlencoded",
            }
        })
        const user = await followup.json()
        const {user_id,team_id} = user.team_user
        
        await updateDoc(doc.ref,{user_id,team_id,timeStamp: serverTimestamp()})
    }catch(e:any){
        if (e.message === "NEXT_REDIRECT") throw e;
        return Response.json({err:e.toString()})
    }finally{
        redirect('/dashboard?nonce=asfHsdfkaRUiwer=')
    }
    
}

