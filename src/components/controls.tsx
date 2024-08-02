import React,{Dispatch,SetStateAction} from "react";
import { Rows,Columns,Box,Pill,Column,Badge } from "@canva/app-ui-kit";

import AttentionControls from "./modules/attention";
import EnhancementControls from "./modules/enhancement";
import StylizationControls from "./modules/stylization";
import PrivacyControls from "./modules/privacy";

export type ControlType = 'attention' | 'enhancement' | 'stylization' | 'privacy'
export type SelectionType = EnhancementSelection|PrivacySelection|AttentionSelection|StylizationSelection|undefined
export enum ControlSelection{
    attention , enhancement , stylization , privacy
}
export enum EnhancementSelection{
    quality,composite,privacy
}
export enum AttentionSelection{
    rearrangement,additive
}
export enum StylizationSelection{
    cartoonize,beautify,dollify
}
export enum PrivacySelection{
    redact,obfuscate
}

export interface ControlProps{
    type: ControlType;
    action?: EnhancementSelection|PrivacySelection|AttentionSelection|StylizationSelection|undefined;
    setAction: Dispatch<SetStateAction<SelectionType>>;
}

export const Controls = (props:ControlProps)=>{
    return (<Box border="low" borderRadius="large" paddingY="1u" paddingX="2u" alignItems="center">
        <SelectedControl type={props.type} action={props.action} setAction={props.setAction}></SelectedControl>
    </Box>)
}

function SelectedControl(props:ControlProps){
    if(props.type == 'attention'){
        return <AttentionControls type={props.type} action={props.action} setAction={props.setAction}></AttentionControls>
    }else if (props.type == 'enhancement'){
        return <EnhancementControls type={props.type} action={props.action} setAction={props.setAction}></EnhancementControls>
    }else if (props.type == 'stylization'){
        return <StylizationControls type={props.type} action={props.action} setAction={props.setAction}></StylizationControls>
    }else if(props.type == 'privacy'){
        return <PrivacyControls type={props.type} action={props.action} setAction={props.setAction}></PrivacyControls>
    }
}


