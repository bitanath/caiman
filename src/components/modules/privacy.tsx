import { Column,Columns,Badge,Pill } from "@canva/app-ui-kit"
import { PrivacySelection,ControlProps } from "../controls"
import { TransparencyIcon,TrashIcon } from "@canva/app-ui-kit"

import { useSelection } from "src/hooks/selection"

export default function PrivacyControls(props:ControlProps){

    const handleClick = (value:PrivacySelection)=>{
        if(props.action == value){
            props.setAction(undefined) //TODO if selected element is clicked again, simply unselect it
        }else{
            props.setAction(value)
        }
    }
    return (
        <Columns spacing="2u" align="center">
              <Column width="content">
                <Badge
                    ariaLabel="Blurs out portions of image or rearranges letters on text in case there is sensitive PII on them"
                    shape="circle"
                    text="i"
                    tone="info"
                    wrapInset="-0.5u"
                    tooltipLabel="Blurs out portions of image or rearranges letters on text in case there is sensitive PII on them"
                  >
                    <Pill
                      role="switch"
                      text="Obfuscate"
                      selected={props.action == PrivacySelection.obfuscate}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(PrivacySelection.obfuscate)}
                      end={<TransparencyIcon></TransparencyIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="Removes images or text in which there is sensitive PII"
                    shape="circle"
                    text="i"
                    tone="critical"
                    wrapInset="-0.5u"
                    tooltipLabel="Removes images or text in which there is sensitive PII"
                  >
                    <Pill
                      role="switch"
                      text="Redact"
                      selected={props.action == PrivacySelection.redact}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(PrivacySelection.redact)}
                      end={<TrashIcon></TrashIcon>}
                    />
                </Badge>
              </Column>
        </Columns>
    )
}
