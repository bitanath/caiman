"use client"

import { ArrowRight, Merge } from "lucide-react"
import Link from "next/link";

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { TextureButton } from "@/components/lib/texture-button"
import {
  TextureCardContent,
  TextureCardFooter,
  TextureCardHeader,
  TextureCardStyled,
  TextureCardTitle,
  TextureSeparator,
} from "@/components/lib/texture-card"

export function ResetComponent() {
  return (
    <div className="flex items-center justify-center py-4">
      <div className="dark:bg-stone-950  h-full rounded-md">
        <div className=" items-start justify-center gap-6 rounded-lg p-2 md:p-8 grid grid-cols-1 ">
          <div className="col-span-1 grid items-start gap-6 lg:col-span-1">
            <div>
              <TextureCardStyled>
                <TextureCardHeader className="flex flex-col gap-1 items-center justify-center p-4">
                  <div className="p-3 bg-neutral-950 rounded-full mb-3">
                    <Merge className="h-7 w-7 stroke-neutral-200" />
                  </div>
                  <TextureCardTitle>Create your account</TextureCardTitle>
                  <p className="text-center">
                    Welcome! Please enter your email below to reset your password.
                  </p>
                </TextureCardHeader>
                <TextureSeparator />
                <TextureCardContent>
                  <form className="flex flex-col gap-6">
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        required
                        className="w-full px-4 py-2 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white/80 dark:bg-neutral-800/80 placeholder-neutral-400 dark:placeholder-neutral-500"
                      />
                    </div>
                  </form>
                </TextureCardContent>
                <TextureSeparator />
                <TextureCardFooter className="border-b rounded-b-sm">
                  <TextureButton variant="accent" className="w-full">
                    <div className="flex gap-1 items-center justify-center">
                      Reset Password
                      <ArrowRight className="h-4 w-4 text-neutral-50 mt-[1px]" />
                    </div>
                  </TextureButton>
                </TextureCardFooter>

                <div className="dark:bg-neutral-800 bg-stone-100 pt-px rounded-b-[20px] overflow-hidden ">
                  <div className="flex flex-col items-center justify-center">
                    <div className="py-2 px-2">
                      <div className="text-center text-sm">
                        Don&apos;t have an account ?{" "}
                        <Link href="/signup"><span className="text-primary">Sign up here</span></Link>
                      </div>
                    </div>
                  </div>
                  <TextureSeparator />
                  <div className="flex flex-col items-center justify-center ">
                    <div className="py-2 px-2">
                      <div className="text-center text-xs ">
                        Go back to <Link href="/login"> Login Here</Link>
                      </div>
                    </div>
                  </div>
                </div>
              </TextureCardStyled>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
