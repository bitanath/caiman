import { getFirestore, collection, query, where, getDocs } from 'firebase/firestore'
import { initializeServerApp } from 'firebase/app'
import { clientConfig } from '@/config'

const app = initializeServerApp(clientConfig,{})
 
export const dynamic = 'force-dynamic'
export async function GET(request: Request) {
  const db = getFirestore(app)
  
  console.log("Got firestore db",db)
  
  const q = query(collection(db,"designs"),where("appUserId","==","fsdfsdfsdf"))
  const results = await getDocs(q)
  const resultDocs = results.docs.map(d=>d.get("uid"))
  return Response.json({ "teri":resultDocs })
}

