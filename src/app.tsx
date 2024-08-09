import React,{useState,useEffect,useCallback} from "react";

import styles from "styles/components.css";
import Spacer from "./components/spacer";
import Toast from "./components/toast";
import Interface from "./components/interface";

import { ImageCard, Button, Rows, Columns, Box, Text, MultilineInput, Title, Carousel, Badge, SegmentedControl, Column, Link } from "@canva/app-ui-kit";
import { OpenInNewIcon, EyeIcon, UndoIcon, WordCountDecorator, TextPlaceholder, TitlePlaceholder, Placeholder, ReloadIcon } from "@canva/app-ui-kit";
import { addNativeElement,getDesignToken, requestExport } from "@canva/design";
import { requestOpenExternalUrl,appProcess } from "@canva/platform";
import { config } from "./settings";
import { fetchRetry } from "./utils";

import { auth } from "@canva/user";


export const App = () => {
  
  const [loading,setLoading] = useState(true)
  const [design,setDesign] = useState(null)
  const [user,setUser] = useState(null)
  const [linkage,setLinkage] = useState<string|null>(null)
  const [reload,setReload] = useState(false)
  
  const [alert,setAlert] = useState<string|undefined>(undefined)
  const [level,setAlertLevel] = useState<'critical'|'warn' | 'positive' | 'info'>('info')

  useEffect(()=>{
    getAuth()
  },[])

  const getAuth = ()=>{
    console.log(user,linkage)
    auth.getCanvaUserToken().then(userToken=>{
      getDesignToken().then(took=>{
        const designToken = took.token
        return fetch(`${config.apiUrl}/api/user/${designToken}`,{
          method: "GET",
          mode: 'cors',
          headers: {
            Authorization: `Bearer ${userToken}`,
            "Content-Type": "application/json",
            "Accept": "application/json"
          }
        })
      }).then(done=>{
        return done.json()
      }).then(json=>{
        setLoading(false)
        if(json.error){
          showAlert(json.error,'warn')
        }
        console.log("Got response json ",json)
        const {userId,designId,brandId,uuId} = json
        setUser(userId)
        setDesign(designId)
        setLinkage(uuId)
      })
    }).catch(err=>{
      showAlert("Please connect to canva in order to continue",'critical')
    })
  }

  const showAlert = (message:string,level:'critical' | 'warn' | 'positive' | 'info'='info')=>{
    setAlertLevel(level)
    setAlert(message)
  }

  async function handleClick() {
    const response = await requestOpenExternalUrl({
      url: `${config.apiUrl}/login?canva=${user}`
    });

    if(response && response.status == "COMPLETED"){
      setReload(true)
    } 
  }

  if(!linkage){
    //Show loading state
    return (
      <div className={styles.scrollContainer}>
        <Toast visible={!!alert} tone={level} onDismiss={()=>setAlert(undefined)}>{alert}</Toast>
        <Spacer padding="0.5u"></Spacer>
        <Rows spacing="0.5u" align="stretch">
            <div style={{height: '280px'}}>
              <Placeholder shape="rectangle"></Placeholder>
            </div>
            <TitlePlaceholder size="large"></TitlePlaceholder>
            <TextPlaceholder size="large"></TextPlaceholder>
            
            <Button variant="primary" loading={loading} onClick={handleClick}>Connect to Canva</Button>
            <Columns spacing="0.5u" align="center">
              <Text size="xsmall" alignment="center">
                {loading ? "Loading Caiman..." : "Link this design to Canva in order to enable Caiman to make suggestions and edits programmatically" }
              </Text>
              <Spacer direction="horizontal" padding="1u"></Spacer>
              <Badge text="?" tone="info" tooltipLabel="Caiman uses AI algorithms to work on the entire design. Link this design in order to enable seamless editing."></Badge>
            </Columns>
            {reload ? <Button variant="secondary" onClick={()=>{getAuth()}} icon={()=><ReloadIcon></ReloadIcon>} iconPosition="start">Reload</Button> : <Spacer padding="0"></Spacer>}
        </Rows>
      </div>
    )
  }else{
    return <Interface setLinkage={setLinkage}></Interface>
  }
  
};
