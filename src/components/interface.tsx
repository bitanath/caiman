import { ImageCard, Button, Rows, Columns, Box, Text, MultilineInput, Title, Carousel, Badge, SegmentedControl, Column, Link } from "@canva/app-ui-kit";
import { OpenInNewIcon, EyeIcon, UndoIcon, WordCountDecorator, TextPlaceholder, TitlePlaceholder, Placeholder } from "@canva/app-ui-kit";

import styles from "styles/components.css";
import Spacer from "./spacer";
import Toast from "./toast";

import { DummyElement,ImageProps } from "./thumb";

import { config } from "../settings";
import { auth } from "@canva/user";
import { addNativeElement,getDesignToken, requestExport } from "@canva/design";

import React,{useState,useEffect,useCallback, Dispatch, SetStateAction} from "react";

export default function Interface({showAlert,level,setAlert,alert}:{
    showAlert:(message:string,level:'critical' | 'warn' | 'positive' | 'info')=>void;
    setAlert: Dispatch<SetStateAction<string|undefined>>
    level:'critical'|'warn' | 'positive' | 'info'
    alert?:string|undefined
}){
    const heatmapImage = "https://img.freepik.com/free-vector/gradient-heat-map-background_23-2149528520.jpg?t=st=1722156716~exp=1722160316~hmac=0390026c4ba6a8059642476127ac95ba3faab20bc4cd38a0cc2be72228db2fa0&w=2000"
    const [images,setImages] = useState([]) //list of image props
    const [currentIndex,setCurrentIndex] = useState(0)

    useEffect(()=>{
        //TODO now query for design here and get all the pages
        fetchAllImages()
    })

    function fetchImageProperties(url:string){
      auth.getCanvaUserToken().then(userToken=>{
        getDesignToken().then(took=>{
          const designToken = took.token
          return fetch(`${config.apiUrl}/api/design/${designToken}`,{
            method: "POST",
            mode: 'cors',
            headers: {
              Authorization: `Bearer ${userToken}`,
              "Content-Type": "application/json",
              "Accept": "application/json"
            },
            body: JSON.stringify({
              "url":url
            })
          })
        }).then(done=>{
          return done.json()
        }).then(json=>{
          console.log("Got response from server",json)
        })
      })
    }

    function fetchAllImages(){
      auth.getCanvaUserToken().then(userToken=>{
        getDesignToken().then(took=>{
          const designToken = took.token
          return fetch(`${config.apiUrl}/api/retrieve/${designToken}`,{
            method: "POST",
            mode: 'cors',
            headers: {
              Authorization: `Bearer ${userToken}`,
              "Content-Type": "application/json",
              "Accept": "application/json"
            },
            body: JSON.stringify({
              "fetch":"all"
            })
          })
        }).then(done=>{
          return done.json()
        }).then(json=>{
          console.log("Got exported designs ",json)
          const {urls}:{urls:[string]} = json
          if(!urls) throw 'Unable to get exported design'
          //TODO handle multi page designs
          const arr = urls.map((url,index)=>{
            console.log(url)
            if(index == currentIndex){
              console.log("Fetching image properties for ",url)
              fetchImageProperties(url)
            }
            return {thumbnailUrl:url,thumbnailHeight:280,index}
          })

        }).catch(err=>{
            showAlert("Error "+err.toString(),"critical")
        })
      })
    }

    return (
        <div className={styles.scrollContainer}>
          <Toast visible={!!alert} tone={level} onDismiss={()=>{setAlert(undefined)}}>{alert}</Toast>
          <Spacer padding="0.5u"></Spacer>
          <Rows spacing="0.5u">
              <Carousel>
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
                
              </Carousel>
              
              <Columns spacing="0.5u" align="center">
                <Text size="xsmall" alignment="center">
                  Bluer tones indicate more attention
                </Text>
                <Spacer direction="horizontal" padding="1u"></Spacer>
                <Badge text="?" tone="info" tooltipLabel="Simulated using an ML Model trained on generic designs. May not be 100% accurate to human representation of attention."></Badge>
              </Columns>
            
          </Rows>
        </div>
      );
}