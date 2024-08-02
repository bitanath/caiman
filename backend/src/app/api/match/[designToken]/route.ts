import * as Functions from './functions' 
export const dynamic = 'force-dynamic'

export async function POST(request: Request,{ params }: { params: { designToken: string } }) {
  try{
    const header = request.headers.get("authorization")
    const userToken = header?.split(" ")[1]
    const {designToken} = params
    const body = await request.json();

    if(!userToken || !designToken) return

    const verifiedDesignToken = await Functions.verifyDesignToken(designToken)
    const verifiedUserToken = await Functions.verifyUserToken(userToken)
    if(!verifiedDesignToken) return Response.json({ "error":"Unable to retrieve Design" },{status:403})
    if(!verifiedUserToken) return Response.json({ "error":"Unable to retrieve User" },{status:403})

    const {designId} = verifiedDesignToken as DesignTokenInterface
    const {brandId,userId} = verifiedUserToken as UserTokenInterface
    if(!designId) return Response.json({ "error":"Unable to retrieve Design Id" },{status:403})
    if(!userId) return Response.json({ "error":"Unable to retrieve Design Id" },{status:403})
    
    
    return Response.json({ designId,brandId,userId},{status:200})
  }catch(e){
    return Response.json({ "error": "Unable to find user or design"},{status:403})
  }
}

interface DesignTokenInterface{
  designId:string;
}

interface UserTokenInterface{
  userId:string;
  brandId:string;
}