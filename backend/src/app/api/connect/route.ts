import type { NextApiRequest, NextApiResponse } from 'next'
import { getFirestore, collection, doc, updateDoc, addDoc, query, where, getDocs } from 'firebase/firestore'
import { initializeServerApp } from 'firebase/app'
import { clientConfig } from '@/config'

const app = initializeServerApp(clientConfig,{})
 
export const dynamic = 'force-dynamic'
export async function GET(request: Request) {
  const db = getFirestore(app)
  
  console.log("Got firestore db",db)
  
  let q = query(collection(db,"designs"),where("appUserId","==","fsdfsdfsdf"))
  
  let results = await getDocs(q)
  let resultDocs = results.docs.map(d=>d.get("uid"))
  return Response.json({ "teri":resultDocs })
}

