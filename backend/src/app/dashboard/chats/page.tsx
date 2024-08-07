import { getTokens } from "next-firebase-auth-edge";
import { cookies } from "next/headers";
import { notFound } from "next/navigation";
import { clientConfig, serverConfig } from "@/config";

import { Input } from "@/components/ui/input"
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion"

export default function Component() {
  return (
    <div className="flex items-center justify-center w-full py-8">
      <div className="w-[80%] rounded-lg bg-background border border-input">
        <div className="flex items-center justify-center my-4">
          <Input
            type="search"
            placeholder="Search messages..."
            className="w-full max-w-2xl rounded-md border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
        <Accordion type="single" collapsible>
          <AccordionItem value="item-1">
            <AccordionTrigger className="flex items-center justify-between bg-card p-4 rounded-md">
              <div className="flex items-center gap-4">
                <img
                  src="/placeholder.svg"
                  alt="Thumbnail"
                  width="64"
                  height="64"
                  className="rounded-md"
                  style={{ aspectRatio: "64/64", objectFit: "cover" }}
                />
                <div className="grid gap-1">
                  <div className="font-medium">John Doe</div>
                  <div className="text-sm text-muted-foreground">5 messages - 2 days ago</div>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="bg-card p-4 rounded-md mt-2 grid gap-4">
              <div className="grid gap-3">
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#55efc4] text-2xl flex items-center justify-center">
                    😁
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">John Doe</div>
                      <div className="text-xs text-muted-foreground">2 days ago</div>
                    </div>
                    <div>
                      <p>Hey, how&apos;s it going?</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#ffeaa7] text-2xl flex items-center justify-center">
                    😎
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Jane Smith</div>
                      <div className="text-xs text-muted-foreground">2 days ago</div>
                    </div>
                    <div>
                      <p>I&apos;m doing great, thanks for asking!</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#fdcb6e] text-2xl flex items-center justify-center">
                    🤠
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">John Doe</div>
                      <div className="text-xs text-muted-foreground">2 days ago</div>
                    </div>
                    <div>
                      <p>Awesome, let&apos;s catch up soon!</p>
                    </div>
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-2">
            <AccordionTrigger className="flex items-center justify-between bg-card p-4 rounded-md">
              <div className="flex items-center gap-4">
                <img
                  src="/placeholder.svg"
                  alt="Thumbnail"
                  width="64"
                  height="64"
                  className="rounded-md"
                  style={{ aspectRatio: "64/64", objectFit: "cover" }}
                />
                <div className="grid gap-1">
                  <div className="font-medium">Jane Smith</div>
                  <div className="text-sm text-muted-foreground">8 messages - 1 week ago</div>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="bg-card p-4 rounded-md mt-2 grid gap-4">
              <div className="grid gap-3">
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#55efc4] text-2xl flex items-center justify-center">
                    😁
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Jane Smith</div>
                      <div className="text-xs text-muted-foreground">1 week ago</div>
                    </div>
                    <div>
                      <p>Hey, did you see the new design?</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#ffeaa7] text-2xl flex items-center justify-center">
                    😎
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">John Doe</div>
                      <div className="text-xs text-muted-foreground">1 week ago</div>
                    </div>
                    <div>
                      <p>Yeah, it looks great! I like the new color scheme.</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#fdcb6e] text-2xl flex items-center justify-center">
                    🤠
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Jane Smith</div>
                      <div className="text-xs text-muted-foreground">1 week ago</div>
                    </div>
                    <div>
                      <p>Awesome, I&apos;ll start implementing it tomorrow.</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#55efc4] text-2xl flex items-center justify-center">
                    😁
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">John Doe</div>
                      <div className="text-xs text-muted-foreground">1 week ago</div>
                    </div>
                    <div>
                      <p>Sounds good, let me know if you need any help!</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#ffeaa7] text-2xl flex items-center justify-center">
                    😎
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Jane Smith</div>
                      <div className="text-xs text-muted-foreground">1 week ago</div>
                    </div>
                    <div>
                      <p>Will do, thanks!</p>
                    </div>
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-3">
            <AccordionTrigger className="flex items-center justify-between bg-card p-4 rounded-md">
              <div className="flex items-center gap-4">
                <img
                  src="/placeholder.svg"
                  alt="Thumbnail"
                  width="64"
                  height="64"
                  className="rounded-md"
                  style={{ aspectRatio: "64/64", objectFit: "cover" }}
                />
                <div className="grid gap-1">
                  <div className="font-medium">Alex Johnson</div>
                  <div className="text-sm text-muted-foreground">3 messages - 3 days ago</div>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="bg-card p-4 rounded-md mt-2 grid gap-4">
              <div className="grid gap-3">
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#55efc4] text-2xl flex items-center justify-center">
                    😁
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Alex Johnson</div>
                      <div className="text-xs text-muted-foreground">3 days ago</div>
                    </div>
                    <div>
                      <p>Hey, did you get my email about the project update?</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#ffeaa7] text-2xl flex items-center justify-center">
                    😎
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">John Doe</div>
                      <div className="text-xs text-muted-foreground">3 days ago</div>
                    </div>
                    <div>
                      <p>Yeah, I just reviewed it. Looks good!</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="rounded-full w-10 h-10 bg-[#fdcb6e] text-2xl flex items-center justify-center">
                    🤠
                  </div>
                  <div className="grid gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">Alex Johnson</div>
                      <div className="text-xs text-muted-foreground">3 days ago</div>
                    </div>
                    <div>
                      <p>Great, I&apos;ll go ahead and implement the changes.</p>
                    </div>
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  )
}