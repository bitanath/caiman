import * as Functions from '@/app/api/utils' 
import { signInWithCustomToken,getAuth } from 'firebase/auth'
import {DesignTokenInterface,UserTokenInterface} from "../../utils"
import {doc,getDoc,getDocs,collection,query,where,getFirestore, updateDoc, setDoc} from 'firebase/firestore'
import {app} from '@/../firebase'
import { tokenize } from '../../utils'
import { GoogleGenerativeAI,FunctionDeclarationSchemaType,HarmCategory,HarmBlockThreshold } from '@google/generative-ai'

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
    const collectionRef = collection(db,"config")
    const queried = query(collectionRef,where("field","==","default"))
    const querySnapshot = await getDocs(queried)
    if(querySnapshot.empty) return Response.json({ "error":" Unable to get configuration correctly " },{status:403})
    const apiKey = querySnapshot.docs[0].get("api_key")
    const {prompt,instruction} = await request.json()

    if(!prompt || !instruction) return Response.json({ "error":"User not linked" },{status:403})

    const genAI = new GoogleGenerativeAI(apiKey)   
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash",
        systemInstruction: "You are a helpful bot by the name of Caiman. Your job is to take user instruction to modify a text prompt. Do not make small talk. \nReturn two values, \n1) A message to the user saying you have modified the image accordingly \n 2) the modified prompt of length 60 words or less",
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
    return Response.json({ messageToUser,promptForModel },{status:200})

  }catch(e:any){
    return Response.json({ "error":e.toString() },{status:403})
  }
}

