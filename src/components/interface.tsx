import { ImageCard, Button, Rows, Columns, Box, Text, MultilineInput, Title, Carousel, Badge, SegmentedControl, Column, Link, Alert, ProgressBar, LoadingIndicator } from "@canva/app-ui-kit";
import { OpenInNewIcon, EyeIcon, UndoIcon, WordCountDecorator, TextPlaceholder, TitlePlaceholder, Placeholder } from "@canva/app-ui-kit";

import styles from "styles/components.css";
import Spacer from "./spacer";
import Toast from "./toast";
import Page from "./page";

import { DummyElement,ImageProps } from "./thumb";

import { config } from "../settings";
import { auth } from "@canva/user";
import { addNativeElement,getDesignToken, requestExport } from "@canva/design";

import React,{useState,useEffect} from "react";

export default function Interface({setLinkage}:{
  setLinkage: React.Dispatch<React.SetStateAction<string|null>>;
}){

    const [images,setImages] = useState<PageInterface[]>([]) //list of image props
    const [errorMessage,setErrorMessage] = useState<string|undefined>(undefined)
    const [lastFetch,setLastFetch] = useState<number>(0)

    useEffect(()=>{
        const currentTime = Date.now()
        if((currentTime - lastFetch) > 60*1000){
          fetchAllImages()
        }
    })

    function fetchAllImages(callback?:()=>void|null){
      console.log("Exporting design")
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
          const {urls}:{urls:[string]} = json
          if(!urls) throw json.error || 'Unable to get exported design'
          //TODO handle multi page designs
          const arr = urls.map((url,index)=>{
            console.log(url)
            return {image:url,height:280,index}
          })
          console.log("Laying out images again")
          setImages(arr)
          setLastFetch(Date.now())

          if(callback){
            return callback()
          }

        }).catch(err=>{
            setErrorMessage("Error Try Reconnecting to Canva from the Caiman website. "+err.toString())
            setLinkage(null)
        })
      })
    }

    return (
        <div className={styles.scrollContainer}>
          <Rows spacing="0.5u">
            {
              images.length < 1 ?
              <Box>
                {errorMessage ? 
                <>
                  <Alert tone="critical">{errorMessage}</Alert>
                  <Spacer padding="0.5u"></Spacer>
                </> : <></>}
                <div style={{height:"280px"}}>
                  <Placeholder shape="rectangle"></Placeholder>
                </div>
                <TitlePlaceholder></TitlePlaceholder>
                <TextPlaceholder></TextPlaceholder>
                <Spacer padding="0.5u"></Spacer>
                <LoadingIndicator size="large"></LoadingIndicator>
                <Text alignment="center" size="small">Fetching designs from Canva</Text>
              </Box>
              : images.length < 2 ?
              <Page image={images[0].image} height={images[0].height} index={images[0].index} globalRefresh={fetchAllImages}></Page>
              : 
              <Carousel>
                {
                  images.map((page,idx)=>{
                    return <Page image={page.image} height={page.height} index={page.index} globalRefresh={fetchAllImages}></Page>
                  })
                }
              </Carousel>
            }
          </Rows>
        </div>
      );
}

interface PageInterface{
  image:string;
  index:number;
  height:number;
}