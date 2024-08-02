import { Column,Columns,Badge,Pill } from "@canva/app-ui-kit"
import { EnhancementSelection,ControlProps } from "../controls"
import { StarFilledIcon,FileTextIcon } from "@canva/app-ui-kit"

import { useSelection } from "src/hooks/selection"

export default function EnhancementControls(props:ControlProps){

    const handleClick = (value:EnhancementSelection)=>{
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
                    ariaLabel="Enhances Image resolution and quality if Image, Faces are auto-enhanced, Text coherence increases on Text"
                    shape="circle"
                    text="i"
                    tone="info"
                    wrapInset="-0.5u"
                    tooltipLabel="Enhances Image resolution and quality if Image, Faces are auto-enhanced, Text coherence increases on Text"
                  >
                    <Pill
                      role="switch"
                      text="Quality"
                      selected={props.action == EnhancementSelection.quality}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(EnhancementSelection.quality)}
                      end={<StarFilledIcon></StarFilledIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="Destructively changes Text or Image to better fit what it is trying to say"
                    shape="circle"
                    text="i"
                    tone="critical"
                    wrapInset="-0.5u"
                    tooltipLabel="Destructively changes Text or Image to better fit what it is trying to say"
                  >
                    <Pill
                      role="switch"
                      text="Content"
                      selected={props.action == EnhancementSelection.privacy}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(EnhancementSelection.privacy)}
                      end={<FileTextIcon></FileTextIcon>}
                    />
                </Badge>
              </Column>
        </Columns>
    )
}
