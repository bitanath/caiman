import { Column,Columns,Badge,Pill } from "@canva/app-ui-kit"
import { AttentionSelection,ControlProps } from "../controls"
import { SortIcon,PlusIcon } from "@canva/app-ui-kit"

import { useSelection } from "src/hooks/selection"

export default function AttentionControls(props:ControlProps){

    const handleClick = (value:AttentionSelection)=>{
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
                    ariaLabel="Rearranges Images or Text in order to make them grab attention a bit more"
                    shape="circle"
                    text="i"
                    tone="contrast"
                    wrapInset="-0.5u"
                    tooltipLabel="Rearranges Images or Text in order to make them grab attention a bit more"
                  >
                    <Pill
                      role="switch"
                      text="Rearrange"
                      selected={props.action == AttentionSelection.rearrangement}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(AttentionSelection.rearrangement)}
                      end={<SortIcon></SortIcon>}
                    />
                </Badge>
              </Column>
              <Column width="content">
                <Badge
                    ariaLabel="Adds Images or Text in order to make it grab more attention while keeping to the original theme"
                    shape="circle"
                    text="i"
                    tone="critical"
                    wrapInset="-0.5u"
                    tooltipLabel="Adds Images or Text in order to make it grab more attention while keeping to the original theme"
                  >
                    <Pill
                      role="switch"
                      text="Add Content"
                      selected={props.action == AttentionSelection.additive}
                      ariaLabel="settings button"
                      onClick={()=>handleClick(AttentionSelection.additive)}
                      end={<PlusIcon></PlusIcon>}
                    />
                </Badge>
              </Column>
        </Columns>
    )
}
