"use client"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, } from "@/components/ui/card";
import { HStack, Spacer } from "@kuma-ui/core";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

export default function AccountInfo({
    displayName,
    email,
    customToken
}:{
    displayName: string;
    email: string;
    customToken: string;
}){
    return (
        <Card className="col-span-4">
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
              <CardDescription>
                View and manage your account details.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Name</Label>
                  <Input id="name" defaultValue={displayName} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={email} disabled />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="username">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    defaultValue={"somelongstringherethatdoesntmakesense"}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password">Confirm Password</Label>
                  <Input
                    id="confirmpassword"
                    type="password"
                    defaultValue={"somelongstringherethatdoesntmakesense"}
                  />
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <HStack>
                <Button>Save Changes</Button>
                <Spacer width={10}></Spacer>
                <Button variant={"destructive"}> Delete Account </Button>
              </HStack>
            </CardFooter>
          </Card>
    )
}