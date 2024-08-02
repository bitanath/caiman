import * as admin from 'firebase-admin'
import { serverApp } from '@/../firebase'

const db = admin.firestore(serverApp)
 
export const dynamic = 'force-dynamic'
export async function GET(request: Request) {
  const citiesRef = db.collection('designs');
  const snapshot = await citiesRef.where('appUserId', '==', "fsdfsdfsdf").get();
  snapshot.forEach(doc => {
    console.log(doc.id, '=>', doc.data());
  })
  const items = snapshot.docs.map(d=>d.data())
  return Response.json({ "teri": process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID, "maka":"bhosda",items })
}

