import { Rows,Columns,Box } from "@canva/app-ui-kit";
import { BoxProps } from "@canva/app-ui-kit/dist/cjs/ui/apps/developing/ui_kit/components/box/box";

interface SpacerProps extends BoxProps {
    padding: "0" | "0.5u" | "1u" | "1.5u" | "2u" | "3u" | "4u" | "6u" | "8u" | "12u" | undefined;
    direction?: "horizontal" | "vertical";
}


const Spacer = (props:SpacerProps)=>{
    if(props.direction == "horizontal"){
        return (
            <Columns spacing="1u">
                <Box paddingX={props.padding}></Box>
            </Columns>
        )
    }else{
        return (
            <Rows spacing="1u">
                <Box paddingY={props.padding}></Box>
            </Rows>
        )
    }
    
}

export default Spacer
