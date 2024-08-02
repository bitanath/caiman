import React,{useState,useEffect,useCallback} from "react";

import styles from "styles/components.css";
import Spacer from "./components/spacer";
import Toast from "./components/toast";

import { ImageCard, Button, Rows, Columns, Box, Text, MultilineInput, Title, Pill, Badge, SegmentedControl, Column } from "@canva/app-ui-kit";
import { OpenInNewIcon, EyeIcon, UndoIcon, WordCountDecorator } from "@canva/app-ui-kit";
import { addNativeElement,getDesignToken } from "@canva/design";
import { Controls,ControlType,SelectionType} from "./components/controls";

import { auth } from "@canva/user";

const controls = [
  { "label": "Attend", "value": "attention" },
  { "label": "Enhance", "value": "enhancement" },
  { "label": "Stylize", "value": "stylization" },
  { "label": "Privacy", "value": "privacy" }
]

export const App = () => {
  const [heatmapImage,setHeatmapImage] = useState("https://img.freepik.com/free-vector/gradient-heat-map-background_23-2149528520.jpg?t=st=1722156716~exp=1722160316~hmac=0390026c4ba6a8059642476127ac95ba3faab20bc4cd38a0cc2be72228db2fa0&w=2000")
  const [designId,setDesignId] = useState(null)
  const [alert,setAlert] = useState<string|undefined>(undefined)
  const [level,setAlertLevel] = useState<'warn' | 'positive' | 'info'>('info')

  const [control,setControl] = useState<ControlType>("enhancement")
  const [action,setAction] = useState<SelectionType>(undefined)

  useEffect(()=>{
    // window.parent.addEventListener("scroll",e=>{
    //   console.log("And the window is scrolling")
    // })
    auth.getCanvaUserToken().then(e=>{
      console.log("Got this user token",e)
    })
    
    // console.log("Parent",window.parent.document.querySelectorAll('div[data-page-id]'))
    console.log("Terimaki")
  },[])

  const showAlert = (message:string,level:'warn' | 'positive' | 'info'='info')=>{
    setAlertLevel(level)
    setAlert(message)
  }

  const showControls = (selected:string)=>{
    try{
      let value = selected as ControlType
      setControl(value) //set a control tab
      setAction(undefined) //set an action to undefined initially, wait for the user to choose
    }catch(e){
      setAlertLevel(level)
      setAlert("Error Type Mismatch: Controls defined do not match selected control")
    }
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
        <Spacer padding="0.5u"></Spacer>
        <Box border="low" className="hr"></Box>
        <Spacer padding="0.5u"></Spacer>
        <SegmentedControl
          defaultValue={control}
          options={controls}
          onChange={showControls}
        />
        <Controls type={control} action={action} setAction={setAction}></Controls>
        
        {/* <Box border="low" borderRadius="large" paddingY="1u" paddingX="2u">
          <Columns spacing="2u" align="spaceBetween">
              <Column width="content">
                <Badge
                    ariaLabel="2 new settings modified"
                    shape="circle"
                    text="2"
                    tone="info"
                    wrapInset="-0.5u"
                  >
                    <Pill
                      text="Fadar"
                      ariaLabel="settings button"
                      end={<CogIcon></CogIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="2 new settings modified"
                    shape="circle"
                    text="2"
                    tone="info"
                    wrapInset="-0.5u"
                  >
                    <Button
                      ariaLabel="settings button"
                      icon={() => {return <CogIcon />}}
                      variant="secondary"
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="2 new settings modified"
                    shape="circle"
                    text="2"
                    tone="info"
                    wrapInset="-0.5u"
                  >
                    <Button
                      ariaLabel="settings button"
                      icon={() => {return <CogIcon />}}
                      variant="secondary"
                    />
                </Badge>
              </Column>
              
          </Columns>
        </Box> */}
        
        <MultilineInput
          footer={<WordCountDecorator max={10} />}
          onChange={() => {}}
          placeholder="(Optional) Add a short prompt to assist the AI in making the alternative"
        />
        <Spacer padding="0.5u"></Spacer>
        <Box border="low" className="hr"></Box>
        <Spacer padding="0.5u"></Spacer>
        <Button variant="primary" onClick={()=>showAlert("Nothing to Preview")} icon={()=>{return <EyeIcon />}} tooltipLabel="Previews the current setting" iconPosition="start" loading={false} stretch>
              Preview
        </Button>
        <Button variant="secondary" onClick={()=>showAlert("Nothing to Undo")} icon={()=>{return <UndoIcon />}} iconPosition="start" stretch>
              Rollback
        </Button>
        <Spacer padding="0.5u"></Spacer>
        <Columns spacing="0.5u">
          <Column width="content">
            <Button variant="secondary" onClick={()=>showAlert("No Alternates","warn")} icon={()=>{return <OpenInNewIcon />}} iconPosition="end">
              View Alternates
            </Button>
          </Column>

          <Spacer direction="horizontal" padding="1u"></Spacer>

          <Column width="content">
            <Badge text="v2.4" tone="neutral" tooltipLabel="Create an alternate for the next bump in version to v2.4" wrapInset="-0.5u">
              <Button variant="primary" onClick={()=>showAlert("Not implemented yet","positive")}>
                Create Alternate
              </Button>
            </Badge>
          </Column>
        </Columns>
        
      </Rows>
    </div>
  );
};
