import { ImageCard, Button, Rows, Columns, Box, Text, Pill, MultilineInput, Title, TypographyCard, Badge, SegmentedControl, Column, Link, LoadingIndicator, EyeIcon, TransparencyIcon, ReloadIcon } from "@canva/app-ui-kit";
import { OpenInNewIcon, LightBulbIcon, WordCountDecorator, TextPlaceholder, TitlePlaceholder, Placeholder, ExportIcon } from "@canva/app-ui-kit";

import styles from "styles/components.css";
import Spacer from "./spacer";
import Toast from "./toast";

import { config } from "../settings";
import { auth } from "@canva/user";
import { addNativeElement,getDesignToken, overlay, getCurrentPageContext } from "@canva/design";
import { upload } from "@canva/asset";

import React,{useState,useEffect} from "react"

export default function Page({image,height,index,globalRefresh}:{
    image:string;
    index:number;
    height:number;
    globalRefresh: (callback:()=>void)=>void;
}){

    const [alert,setAlert] = useState<string|null>(null)
    const [loading,setLoading] = useState(false)
    const [showHeatmap,setShowHeatmap] = useState(false)
    const [showOriginal,setShowOriginal] = useState(true)

    const [heatmap,setHeatmap] = useState<string|null>(null)
    const [alternate,setAlternate] = useState(null)
    const [altmap,setAltmap] = useState(null)
    const [background,setBackground] = useState<string|null>(null)
    const [textlayer,setTextlayer] = useState<string|null>(null)
    
    const [level,setLevel] = useState<'critical'|'warn' | 'positive' | 'info'>('warn')

    const [message,setMessage] = useState<string|undefined>(undefined)
    const [instruction,setInstruction] = useState<string|undefined>(undefined)
    const [reply,setReply] = useState<string|undefined>(undefined)
    const [prompt,setPrompt] = useState<string|undefined>(undefined)

    const alternateDesign = async (message:string)=>{
        try{
            setLoading(true)
            const timeout1 = setTimeout(time=>{
                console.log("Setting first warning")
                setLevel("info")
                setAlert("Taking a bit longer to fetch...")
            },6000)

            const timeout2 = setTimeout(time=>{
                console.log("Setting first warning")
                setLevel("warn")
                setAlert("Don't navigate away I'm still on it...")
            },18000)

            const timeout3 = setTimeout(time=>{
                console.log("Setting second warning")
                setLevel("positive")
                setAlert("Hold on I'm almost done with this...")
            },30000)

            setMessage("")
            setInstruction("🤓 "+message)
            const userToken =  await auth.getCanvaUserToken()
            const otherToken = await getDesignToken()
            const designToken = otherToken.token
            console.log("Getting Caiman Alternates ",message,designToken)
            const serverResponse = await fetch(`${config.apiUrl}/api/alternate/${designToken}`,{
                            method: "POST",
                            mode: 'cors',
                            headers: {
                            Authorization: `Bearer ${userToken}`,
                                "Content-Type": "application/json",
                                "Accept": "application/json"
                            },
                            body: JSON.stringify({
                                "prompt":prompt,
                                "instruction":message,
                                "url":image
                            })
                        })
            
            
            
            const json = await serverResponse.json()
            if(json.error) throw ": Server Failed to Run Prompt"
            setReply("🤖 "+json.message)
            if(json.promptused) setPrompt(json.promptused)
            if(json.alternateUrl) setAlternate(json.alternateUrl)
            if(json.originalmapUrl) setHeatmap(json.originalmapUrl)
            if(json.alternatemapUrl) setAltmap(json.alternatemapUrl)
            if(json.textlayerUrl) setTextlayer(json.textlayerUrl)
            if(json.backgroundUrl) setBackground(json.backgroundUrl)
            setLoading(false)
            setAlert(null)
            if(json.alternateUrl) setShowOriginal(false)

            clearTimeout(timeout1)
            clearTimeout(timeout2)
            clearTimeout(timeout3)

        }catch(e:any){
            console.log("Got error while trying to fetch",e)
            setReply("🤖 Errored out ❌")
            setLevel("critical")
            setAlert("Errored out while trying to fetch: "+e.toString())
            setLoading(false)
        }
    }

    async function addToDesign() {
        // Upload an image
        if(!textlayer || !background) return
        const context = await getCurrentPageContext();
        const dimensions = context.dimensions
        if(!dimensions) return
        console.log("Got dimensions ",dimensions)
        const textoverlay = await upload({
          type: "IMAGE",
          mimeType: "image/png",
          url: textlayer!,
          thumbnailUrl:textlayer!,
        });
        const bgoverlay = await upload({
          type: "IMAGE",
          mimeType: "image/jpeg",
          url: background!,
          thumbnailUrl:background!,
        });
        

        await addNativeElement({
            type: "GROUP",
            children:[
                {
                    type: "IMAGE",
                    ref: bgoverlay.ref,
                    width: dimensions!.width,
                    height: dimensions!.height,
                    top: 0,
                    left: 0,
                },
                {
                    type: "IMAGE",
                    ref: textoverlay.ref,
                    width: dimensions!.width,
                    height: dimensions!.height,
                    top: 0,
                    left: 0,
                },
            ]
        })
    
        
    }

    return (
        <div className={styles.scrollContainer}>
            {
                !!alert ? <div>
                <Toast visible={true} tone={level} onDismiss={()=>{setAlert(null)}}>{alert}</Toast>
                <Spacer padding="0.5u"></Spacer>
                </div>:
                <></>
            }
            <Rows spacing="0.5u">
                {showOriginal ? showHeatmap ?
                <ImageCard
                alt="Not Yet Loaded: Heatmap Describing Attention"
                ariaLabel="Heatmap by visitor attention"
                borderRadius="standard"
                thumbnailHeight={height}
                bottomEnd={<Badge text={"Page: "+(index+1)} tooltipLabel="Heatmap of viewer attention simulated using maching learning" tone="contrast"/>}
                bottomEndVisibility="always"
                loading={loading}
                thumbnailUrl= {heatmap!}
              /> 
                : 
                <ImageCard
                  alt="The original design image to show"
                  ariaLabel="The original design image to show"
                  borderRadius="standard"
                  bottomEnd={<Badge text={"Page: "+(index+1)}  tooltipLabel="Send messages to Caiman to modify or critique this design" tone="contrast"/>}
                  bottomEndVisibility="always"
                  thumbnailUrl= {image}
                  thumbnailHeight={height}
                /> 
                : 
                showHeatmap ?
                <ImageCard
                alt="Not Yet Loaded: Alternative AI Generated Heatmap Describing Attention"
                ariaLabel="Alternative image dreamed up using AI"
                borderRadius="standard"
                bottomEnd={<Badge text={"Page: "+(index+1)}  tooltipLabel="A Heatmap describing attention on the dreamed up alternative" tone="contrast"/>}
                bottomEndVisibility="always"
                onClick={() => {}}
                loading={loading}
                thumbnailUrl= {altmap!}
                thumbnailHeight={height}
              /> :
                <ImageCard
                alt="Not Yet Loaded: Alternative AI Generated Image"
                ariaLabel="Alternative image dreamed up using AI"
                borderRadius="standard"
                bottomEnd={<Badge text={"Page: "+(index+1)}  tooltipLabel="Alternative Dreamed up using AI. Click and Drag the image into your design." tone="contrast"/>}
                bottomEndVisibility="always"
                onClick={async () => {
                    console.log("Clicked on the design")
                    await addToDesign()
                }}
                
                loading={loading}
                thumbnailUrl= {alternate!}
                thumbnailHeight={height}
              /> }
                <Spacer padding="0.5u"></Spacer>
                <Columns spacing="0.5u" align="center">
                    <Pill text="Original" role="switch" selected={showOriginal} disabled={!alternate} onClick={()=>{setShowOriginal(!showOriginal)}} start={<EyeIcon></EyeIcon>}></Pill>
                    <Spacer padding="2u" direction="horizontal"></Spacer>
                    <Pill text="Heatmap" role="switch" selected={showHeatmap} disabled={!heatmap} onClick={()=>{setShowHeatmap(!showHeatmap)}} end={<TransparencyIcon></TransparencyIcon>}></Pill>
                </Columns>
                <Spacer padding="0.5u"></Spacer>
                <Box id="messageTxt" borderRadius="large" alignItems="center" padding="2u" background="neutralLow">
                    {loading ? 
                    <Columns spacing="0.5u">
                        <LoadingIndicator size="small"></LoadingIndicator>
                        <Text size="small">
                            Fetching a response...
                        </Text>
                    </Columns>
                    
                    :
                    <Text size="small">
                        {reply || "🤖 Hi, I'm Caiman, I can critique designs and show you new ideas. Send me a message to get started!"}
                    </Text>}
                </Box>
                
                <Spacer padding="0.5u"></Spacer>
                <TypographyCard ariaLabel="Message sent by Caiman" onClick={async () => {
                    await alternateDesign(instruction?.replace("🤓 ","") || "Hey, how goes it?")
                }} loading={loading} >
                    <Text size="small">
                        {instruction || "🤓 Hey, how goes it?"}
                    </Text>
                </TypographyCard>
                <Spacer padding="0.5u"></Spacer>
                <MultilineInput
                    minRows={2}
                    maxRows={3}
                    footer={<WordCountDecorator max={20} />}
                    onChange={(value:string) => {setMessage(value);}}
                    placeholder="Send your message here"
                    value={message}
                />
                <Spacer padding="0.5u"></Spacer>
                <Button variant="primary" loading={loading} icon={()=><LightBulbIcon></LightBulbIcon>} iconPosition="end" onClick={async ()=>{
                    await alternateDesign(message||"no message")
                }}>Feed the Caiman</Button>
                <Spacer padding="0.5u"></Spacer>
                <Button variant="secondary" loading={loading} icon={()=><ReloadIcon></ReloadIcon>} iconPosition="end" onClick={async ()=>{
                    setAlert("Trying to fetch the latest from Canva")
                    setLevel("info")
                    setTimeout(time=>{
                        setAlert("Please Ensure You Saved Manually")
                        setLevel("warn")
                    },2000)
                    setTimeout(time=>{
                        setAlert("Hang on, almost done!")
                        setLevel("positive")
                    },5000)
                    globalRefresh(()=>{
                        console.log("Finished refresh")
                        setHeatmap(null)
                        setAlert(null)
                    })
                }}>Refresh the Design</Button>
            </Rows>
        </div>
    )
}