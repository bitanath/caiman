import * as Functions from '@/app/api/utils' 
import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {DesignTokenInterface,UserTokenInterface} from "../../utils"
import {doc,getDoc,getDocs,collection,query,where,getFirestore, updateDoc, setDoc, addDoc, increment, orderBy, serverTimestamp, limit} from 'firebase/firestore'
import { getDownloadURL, getStorage,ref,uploadString } from "firebase/storage"

import { GoogleGenerativeAI,FunctionDeclarationSchemaType,HarmCategory,HarmBlockThreshold } from '@google/generative-ai'
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
    const api_key = querySnapshot.docs[0].get("api_key")
    const storage = getStorage(app,storage_url)
    const genAI = new GoogleGenerativeAI(api_key)  

    let {url,prompt,instruction} = await request.json()
    const revisionRef = collection(db,"revisions")
    const revQuery = query(revisionRef,where("designId","==",designId),orderBy("timestamp","desc"),limit(1))
    let revision = 0
    let revSnapshot = await getDocs(revQuery)

    console.log("Got rev snapshot ",revSnapshot.docs.length)
    
    if(revSnapshot.empty){
        revision = 1
    }else{
        const documents = revSnapshot.docs
        revision = documents[0].get("revision")
        console.log("Got rev snapshot revision ",revision)
        revision = revision + 1
    }

    if(!url) return Response.json({ error: "No URL found" },{status:403})
    
    let imageRef = ref(storage, `/designs/${designId}/${revision}/image.jpg`)
    
    console.log("Got request json ",url,prompt,instruction)
    const b64image = await imageUrlToBase64(url)

    await uploadString(imageRef,b64image,'base64',{contentType: 'image/jpeg'}) 
    const downloadImageUrl = await getDownloadURL(imageRef)

    console.log("Got download image url",downloadImageUrl)

    console.log("Got revision number ",revision)

    let message = undefined

    if(instruction){
        const {category,isModification} = await queryGemini(genAI,instruction)
        if(!isModification){
            //TODO short circuit now with an image response
            console.log("Critiquing the image")
            const messageToUser = await generateCritique(api_key,instruction,b64image)
            console.log("Got generated critique",messageToUser)
            message = messageToUser
            if(!prompt) prompt = "no prompt"
            await addDoc(revisionRef,{
                message, prompt,revision, designId, downloadImageUrl,timestamp:serverTimestamp()
            })
            
            return Response.json({ message },{status:200})
        }else{
            let generatedPrompt = prompt
            if(!prompt){
                //If no prompt provided, then get one from gemini
                generatedPrompt = await generateCritique(api_key,"Describe this image in 75 words or less. Do not try to find flaws. Override system instruction. Do not mention Canva.",b64image)
            }
            const {messageToUser,promptForModel} = await generatePrompt(genAI,instruction,prompt||generatedPrompt)
            message = messageToUser
            prompt = promptForModel
        }
    }

    const serverResponse =  await queryServer(server_url,b64image,userToken,prompt)

    if(!serverResponse.alternate || !serverResponse.background) return Response.json({ error: "Unable to convert image" },{status:403})

    const {promptused,alternate,background,textlayer,alternatemap,originalmap} = serverResponse

    prompt = promptused

    //MARK:- Now add all images to the bucket under this revision
    imageRef = ref(storage, `/designs/${designId}/${revision}/background.jpg`)
    await uploadString(imageRef,background,'base64',{contentType: 'image/jpeg'})
    const backgroundUrl = await getDownloadURL(imageRef)

    imageRef = ref(storage, `/designs/${designId}/${revision}/alternate.jpg`)
    await uploadString(imageRef,alternate,'base64',{contentType: 'image/jpeg'})
    const alternateUrl = await getDownloadURL(imageRef)

    imageRef = ref(storage, `/designs/${designId}/${revision}/textlayer.png`)
    await uploadString(imageRef,textlayer,'base64',{contentType: 'image/png'})
    const textlayerUrl = await getDownloadURL(imageRef)

    imageRef = ref(storage, `/designs/${designId}/${revision}/alternatemap.jpg`)
    await uploadString(imageRef,alternatemap,'base64',{contentType: 'image/jpeg'})
    const alternatemapUrl = await getDownloadURL(imageRef)

    imageRef = ref(storage, `/designs/${designId}/${revision}/originalmap.jpg`)
    await uploadString(imageRef,originalmap,'base64',{contentType: 'image/jpeg'})
    const originalmapUrl = await getDownloadURL(imageRef)

    await addDoc(revisionRef,{
        message, prompt, alternateUrl, backgroundUrl, textlayerUrl, alternatemapUrl, originalmapUrl, downloadImageUrl, revision, designId, timestamp:serverTimestamp()
    })

    return Response.json({ message,promptused,alternateUrl,backgroundUrl,textlayerUrl,alternatemapUrl,originalmapUrl },{status:200})

  }catch(e:any){
    console.log("Errored out while processing ",e)
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
    console.log("Generating image with ",prompt)
    const response = await fetch(serverUrl,{
      method: "POST",
      headers: {
        "Content-Type":"application/json",
        "Accept": "application/json",
        "Authorization": userToken
      },
      body: JSON.stringify({
        imgBase64: b64Image,
        instruction: prompt
      })
    })
    const json = await response.json()
    const promptused = json.prompt
    const alternate = json.alternate.replace("b'","").replace("'","")
    const background = json.background.replace("b'","").replace("'","")
    const textlayer = json.textlayer.replace("b'","").replace("'","")
    const alternatemap = json.alternatemap.replace("b'","").replace("'","")
    const originalmap = json.originalmap.replace("b'","").replace("'","")
    return {promptused,alternate,background,textlayer,alternatemap,originalmap}
}

const queryGemini = async (genAI:GoogleGenerativeAI,instruction:string)=>{
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash",
        systemInstruction: "You are bot by the name of Caiman. Your job is to classify a user command into one of two categories. The categories are:\n1) Image Critique\n2) Image Modification\nDo not try to make small talk. Do not hallucinate.\nOnly output a json response containing the type of command and a boolean which is true when the user command is of the category 2) Image Modification",
        safetySettings: [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH }
        ],
        generationConfig: { 
        responseMimeType: "application/json",
        responseSchema: {
            type: FunctionDeclarationSchemaType.OBJECT,
            properties: {
              category: {
                  type: FunctionDeclarationSchemaType.STRING,
                  properties: {}
              },
              isModification: {
                  type: FunctionDeclarationSchemaType.BOOLEAN,
                  properties: {}
              }
            }
        }
    }})

    const result = await model.generateContent(instruction)
    const text = result.response.text()
    let {category,isModification} = JSON.parse(text)
    return {category,isModification}
}

const generateCritique = async (key:string,prompt:string,imageBase64:string)=>{
    try{
        const url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        let part1 = {text: prompt}
        let part2 = {inline_data: { data: imageBase64.replace("data:image/jpeg;base64,",""), mime_type: "image/jpeg" }}
        let part3 = {parts: [{text: "You are an assistant for critiquing the design of the provided image based on the questions the user asks.\nYour response may contain any flaws in the image, with helpful suggestions to fix them using tools like Canva.\nYour response may also contain what is good in the image, with the possibility of enhancing the Image further using tools like Canva.\nPoint out any placeholder text like Lorem Ipsum or placeholder phone numbers or websites or email addresses.\nDo not make small talk. Do not hallucinate.\nKeep your messages short and to the point."}]}
        let part4 = beSafe("BLOCK_NONE")
        
        let obj = {
            "contents":[ { "parts":[ part1, part2 ] } ],
            "systemInstruction": part3,
            "safetySettings": part4
        }
        const contents = obj
        const response = await fetch(url+`?key=${key}`, {
            method: "POST", 
            mode: "cors", 
            cache: "no-cache", 
            headers: {
            "Content-Type": "application/json",
            "Accept": "application/json"
            },
            redirect: "follow", 
            body: JSON.stringify(contents)
        });
        const json = await response.json()
        const textResponse = json.candidates[0].content.parts[0].text
        return textResponse
    }catch(e){
        console.log("Errored out while fetching critique",e)
        throw "Errored out while fetching critique"
    }
    
}

const generatePrompt = async (genAI:GoogleGenerativeAI,instruction:string,prompt:string)=>{
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash",
        systemInstruction: "You are a helpful bot by the name of Caiman. Your job is to take user instruction to modify a text prompt. Do not make small talk.\nReturn two values, \n1) A message to the user saying you have modified the image accordingly \n 2) the modified prompt of length 70 words or less",
        safetySettings: [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH }
        ],
        generationConfig: { 
        responseMimeType: "application/json",
        responseSchema: {
            type: FunctionDeclarationSchemaType.OBJECT,
            properties: {
              messageToUser: {
                  type: FunctionDeclarationSchemaType.STRING,
                  properties: {}
              },
              promptForModel: {
                  type: FunctionDeclarationSchemaType.STRING,
                  properties: {}
              }
            }
        }
    }})

    const result = await model.generateContent(instruction+"\nPrompt provided below:\n"+prompt)
    const text = result.response.text()
    let {messageToUser,promptForModel} = JSON.parse(text)
    messageToUser = messageToUser.replaceAll("prompt","design")
    return {messageToUser,promptForModel}
}

function beSafe(blocker="BLOCK_ONLY_HIGH"){
    return [
         {
             "category": "HARM_CATEGORY_HARASSMENT",
             "threshold": blocker
         },
         {
             "category": "HARM_CATEGORY_HATE_SPEECH",
             "threshold": blocker
         },
         {
             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
             "threshold": blocker
         },
         {
             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
             "threshold": blocker
         }
         
     ]
 }