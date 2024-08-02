import { Alert } from "@canva/app-ui-kit";
import { AlertProps } from "@canva/app-ui-kit/dist/cjs/ui/apps/developing/ui_kit/components/alert/alert";

interface ToastProps extends AlertProps {
    visible:Boolean;
}

const Toast = (props:ToastProps)=>{
    if(props.visible){
        return (<Alert tone={props.tone} onDismiss={props.onDismiss}>{props.children}</Alert>)
    }
}

export default Toast