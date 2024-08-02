import { Column,Columns,Badge,Pill } from "@canva/app-ui-kit"
import { StylizationSelection,ControlProps } from "../controls"
import { LightBulbIcon, PencilIcon, PaintRollerIcon } from "@canva/app-ui-kit"

import { useSelection } from "src/hooks/selection"

export default function StylizationControls(props:ControlProps){

    const handleClick = (value:StylizationSelection)=>{
        if(props.action == value){
            props.setAction(undefined) //TODO if selected element is clicked again, simply unselect it
        }else{
            props.setAction(value)
        }
    }
    return (
        <Columns spacing="0.5u" align="spaceBetween">
              <Column width="content">
                <Badge
                    ariaLabel="Beautify stylizes the image or text (note this destructively replaces image or text)"
                    shape="circle"
                    text="i"
                    tone="contrast"
                    wrapInset="-0.5u"
                    tooltipLabel="Beautify stylizes the image or text (note this destructively replaces image or text)"
                    
                  >
                    <Pill
                      role="switch"
                      text="Beauty"
                      selected={props.action == StylizationSelection.beautify}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(StylizationSelection.beautify)}
                      end={<LightBulbIcon></LightBulbIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="Cartoonizes image or text to make it more appealing to a younger audience"
                    shape="circle"
                    text="i"
                    tone="contrast"
                    wrapInset="-0.5u"
                    tooltipLabel="Cartoonizes image or text to make it more appealing to a younger audience"
                  >
                    <Pill
                      role="switch"
                      text="Toon"
                      selected={props.action == StylizationSelection.cartoonize}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(StylizationSelection.cartoonize)}
                      end={<PencilIcon></PencilIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="Dollify an image or make text sound authentically vintage"
                    shape="circle"
                    text="i"
                    tone="contrast"
                    wrapInset="-0.5u"
                    tooltipLabel="Dollify an image or make text sound authentically vintage"
                  >
                    <Pill
                      role="switch"
                      text="Doll"
                      selected={props.action == StylizationSelection.dollify}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(StylizationSelection.dollify)}
                      end={<PaintRollerIcon></PaintRollerIcon>}
                    />
                </Badge>
              </Column>
        </Columns>
    )
}
