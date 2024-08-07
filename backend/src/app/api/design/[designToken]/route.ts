import * as Functions from '@/app/api/utils' 
import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {DesignTokenInterface,UserTokenInterface} from "../../utils"
import {doc,getDoc,getDocs,collection,query,where,getFirestore, updateDoc, setDoc} from 'firebase/firestore'
import { getDownloadURL, getStorage,ref,uploadString } from "firebase/storage";
import {app} from '@/../firebase'
import { tokenize } from '../../utils'

export const dynamic = 'force-dynamic'
export async function POST(request: Request,{ params }: { params: { designToken: string } }) {
  try{
    //Given a single URL get a heatmap, prompt and alternative image, store heatmap and alternative in storage
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
    if(!designId || !userId) return Response.json({ error: "No User or Design found" },{status:403})
    
    await signInWithCustomToken(getAuth(app),tokenize())
    const db = getFirestore(app)
    const collectionRef = collection(db,"config")
    const queried = query(collectionRef,where("field","==","default"))
    const querySnapshot = await getDocs(queried)
    if(querySnapshot.empty) return Response.json({ "error":" Unable to get configuration correctly " },{status:403})
    const server_url = querySnapshot.docs[0].get("server_url")
    const storage = getStorage(app,'gs://caiman-75133.appspot.com')
    
    const mountainsRef = ref(storage, 'check.jpg')

    const {url,prompt} = await request.json()
    if(!url) return Response.json({ error: "No URL found" },{status:403})
    console.log("Got request json ",url)
    const b64image = await imageUrlToBase64(url)

    uploadString(mountainsRef,b64image,'base64',{contentType: 'image/jpeg'}) //Do not await this value instead let it upload whenever
    const dloadUrl = await getDownloadURL(mountainsRef)

    console.log("Got base 64 image now do something")

    return Response.json({ "maki":"jiho",server_url,url,prompt,b64image },{status:200})

  }catch(e:any){
    return Response.json({ "error":e.toString() },{status:403})
  }
}


const imageUrlToBase64 = async (url:string) => {
    const response = await fetch(url);
    const blob = await response.arrayBuffer();
    const arrBuff = Buffer.from(blob)
    const base64 = arrBuff.toString('base64')
    return base64
};