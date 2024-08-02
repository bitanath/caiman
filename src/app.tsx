import React,{useState,useEffect,useCallback} from "react";

import styles from "styles/components.css";
import Spacer from "./components/spacer";
import Toast from "./components/toast";

import { ImageCard, Button, Rows, Columns, Box, Text, MultilineInput, Title, Pill, Badge, SegmentedControl, Column } from "@canva/app-ui-kit";
import { OpenInNewIcon, EyeIcon, UndoIcon, WordCountDecorator } from "@canva/app-ui-kit";
import { addNativeElement,getDesignToken } from "@canva/design";
import { config } from "./settings";

import { auth } from "@canva/user";


export const App = () => {
  const [heatmapImage,setHeatmapImage] = useState("https://img.freepik.com/free-vector/gradient-heat-map-background_23-2149528520.jpg?t=st=1722156716~exp=1722160316~hmac=0390026c4ba6a8059642476127ac95ba3faab20bc4cd38a0cc2be72228db2fa0&w=2000")
  const [designId,setDesignId] = useState(null)
  const [alert,setAlert] = useState<string|undefined>(undefined)
  const [level,setAlertLevel] = useState<'critical'|'warn' | 'positive' | 'info'>('info')

  useEffect(()=>{
    auth.getCanvaUserToken().then(userToken=>{
      getDesignToken().then(took=>{
        const designToken = took.token
        //Now to call backend
        console.log("Got user token and design token now fetching",userToken == designToken)
        return fetch(`${config.apiUrl}/match/${designToken}`,{
          method: "POST",
          mode: 'cors',
          headers: {
            Authorization: `Bearer ${userToken}`,
            "Content-Type": "application/json",
            "Accept": "application/json"
          },
          body: JSON.stringify({
            "canva":true
          })
        })
      }).then(done=>{
        return done.json()
      }).then(json=>console.log(json))
    }).catch(err=>{
      showAlert("Cannot retrieve user or design",'critical')
    })
  },[])

  const showAlert = (message:string,level:'critical' | 'warn' | 'positive' | 'info'='info')=>{
    setAlertLevel(level)
    setAlert(message)
  }


  async function getDesignTokenFromUI(){
    let token = await getDesignToken()
    console.log("Got from UI ",token)
  }


  return (
    <div className={styles.scrollContainer}>
      <Toast visible={!!alert} tone={level} onDismiss={()=>setAlert(undefined)}>{alert}</Toast>
      <Spacer padding="0.5u"></Spacer>
      <Rows spacing="0.5u">
          <ImageCard
            alt="grass image"
            ariaLabel="Add image to design"
            borderRadius="standard"
            bottomEnd={<Badge text="v2.3" tone="contrast"/>}
            bottomEndVisibility="always"
            onClick={() => {}}
            onDragStart={() => {}}
            thumbnailUrl= {heatmapImage}
          />
          <Columns spacing="0.5u" align="center">
            <Text size="xsmall" alignment="center">
              Bluer tones indicate more attention
            </Text>
            <Spacer direction="horizontal" padding="1u"></Spacer>
            <Badge text="?" tone="info" tooltipLabel="Simulated using an ML Model trained on generic designs. May not be 100% accurate to human representation of attention."></Badge>
          </Columns>
        
        <Button variant="primary" onClick={async e=>{await getDesignTokenFromUI()}}></Button>
      </Rows>
    </div>
  );
};
