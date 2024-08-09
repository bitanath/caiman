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
    const storage_url = querySnapshot.docs[0].get("storage_url")
    const storage = getStorage(app,storage_url)
    
    let imageRef = ref(storage, `/designs/${designId}/image.jpg`)

    const {url,prompt} = await request.json()
    if(!url) return Response.json({ error: "No URL found" },{status:403})
    console.log("Got request json ",url)
    const b64image = await imageUrlToBase64(url)

    await uploadString(imageRef,b64image,'base64',{contentType: 'image/jpeg'}) 
    const downloadImageUrl = await getDownloadURL(imageRef)

    console.log("Got download image url",downloadImageUrl)

    const serverResponse =  await queryServer(server_url,b64image,userToken)
    console.log("Got server response",serverResponse)

    if(!serverResponse.alternate || !serverResponse.background) return Response.json({ error: "Unable to convert image" },{status:403})

    const {alternate,background} = serverResponse
    imageRef = ref(storage, `/designs/${designId}/background.jpg`)
    await uploadString(imageRef,background,'base64',{contentType: 'image/jpeg'})
    const alternateUrl = await getDownloadURL(imageRef)

    imageRef = ref(storage, `/designs/${designId}/alternate.jpg`)
    await uploadString(imageRef,alternate,'base64',{contentType: 'image/jpeg'})
    const backgroundUrl = await getDownloadURL(imageRef)

    return Response.json({ alternateUrl,backgroundUrl },{status:200})

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

const queryServer = async (serverUrl:string,b64Image:string,userToken:string,prompt:string|undefined=undefined)=>{
    const response = await fetch(serverUrl,{
      method: "POST",
      headers: {
        "Content-Type":"application/json",
        "Accept": "application/json",
        "Authorization": userToken
      },
      body: JSON.stringify({
        imgBase64: b64Image,
        prompt: prompt
      })
    })
    const json = await response.json()
    const alternate = json.alternate
    const background = json.background
    return {alternate,background}
}