"use client"

//from cult ui

import { useEffect, useState, ReactNode } from "react"
import { animate, motion, useMotionValue, useTransform } from "framer-motion"

export interface ITypewriterProps {
  delay: number
  texts: string[]
  baseText?: string
}

export function Typewriter({ delay, texts, baseText = "" }: ITypewriterProps) {
  const [animationComplete, setAnimationComplete] = useState(false)
  const count = useMotionValue(0)
  const rounded = useTransform(count, (latest) => Math.round(latest))
  const displayText = useTransform(rounded, (latest) =>
    baseText.slice(0, latest)
  )

  useEffect(() => {
    const controls = animate(count, baseText.length, {
      type: "tween",
      delay,
      duration: 1,
      ease: "easeInOut",
      onComplete: () => setAnimationComplete(true),
    })
    return () => {
      controls.stop && controls.stop()
    }
  }, [count, baseText.length, delay])

  return (
    <span>
      <motion.span>{displayText}</motion.span>
      {animationComplete && (
        <RepeatedTextAnimation texts={texts} delay={delay + 1} />
      )}
      <BlinkingCursor />
    </span>
  )
}

export interface IRepeatedTextAnimationProps {
  delay: number
  texts: string[]
}

const defaultTexts = [
  "quiz page with questions and answers",
  "blog Article Details Page Layout",
  "ecommerce dashboard with a sidebar",
  "ui like platform.openai.com....",
  "buttttton",
  "aop that tracks non-standard split sleep cycles",
  "transparent card to showcase achievements of a user",
]
function RepeatedTextAnimation({
  delay,
  texts = defaultTexts,
}: IRepeatedTextAnimationProps) {
  const textIndex = useMotionValue(0)

  const baseText = useTransform(textIndex, (latest) => texts[latest] || "")
  const count = useMotionValue(0)
  const rounded = useTransform(count, (latest) => Math.round(latest))
  const displayText = useTransform(rounded, (latest) =>
    baseText.get().slice(0, latest)
  )
  const updatedThisRound = useMotionValue(true)

  useEffect(() => {
    const animation = animate(count, 60, {
      type: "tween",
      delay,
      duration: 1,
      ease: "easeIn",
      repeat: Infinity,
      repeatType: "reverse",
      repeatDelay: 1,
      onUpdate(latest) {
        if (updatedThisRound.get() && latest > 0) {
          updatedThisRound.set(false)
        } else if (!updatedThisRound.get() && latest === 0) {
          textIndex.set((textIndex.get() + 1) % texts.length)
          updatedThisRound.set(true)
        }
      },
    })
    return () => {
      animation.stop && animation.stop()
    }
  }, [count, delay, textIndex, texts, updatedThisRound])

  return <motion.span className="inline">{displayText}</motion.span>
}

const cursorVariants = {
  blinking: {
    opacity: [0, 0, 1, 1],
    transition: {
      duration: 1,
      repeat: Infinity,
      repeatDelay: 0,
      ease: "linear",
      times: [0, 0.5, 0.5, 1],
    },
  },
}

function BlinkingCursor() {
  return (
    <motion.div
      variants={cursorVariants}
      animate="blinking"
      className="inline-block h-5 w-[1px] translate-y-1 bg-neutral-900"
    />
  )
}

function IosOgShellCard({ children }: { children: ReactNode }) {
    return (
      <div className="max-w-xs md:max-w-xl md:min-w-80 mx-auto flex flex-col max-h-[400px] min-h-[400px] rounded-lg bg-transparent border px-px pb-px shadow-inner-shadow">
        <div className="p-4 flex flex-col md:px-5">
          <div className="mb-2 text-sm md:text-neutral-500 text-neutral-500">
            Average LLM Interaction
          </div>
          <div className="mb-3 text-xs md:text-sm text-neutral-500">
            Today 11:29
          </div>
          <div className="ml-auto px-4 py-2 mb-3 text-white bg-blue-500 rounded-2xl">
            <span>😺 Hey!</span>
          </div>
          <div className="mr-auto px-4 py-2 mb-3 text-white bg-neutral-700 rounded-2xl">
            <span>🤖 Hello User</span>
          </div>
          <div className="ml-auto px-4 py-2 mb-3 text-white bg-blue-500 rounded-2xl">
            <span>😺 Pls Halp!</span>
          </div>
          <div className="mr-auto px-4 py-2 mb-3 text-white bg-neutral-700 rounded-2xl">
            <span>🤖 Need more context</span>
          </div>
          {children}
          <div className="mt-3 text-xs md:text-sm text-neutral-500">
            Delivered
          </div>
        </div>
      </div>
    )
}

export function ChatCard() {
    const texts = [
        "Make a Google Sheet",
        "Edit my Google Doc",
        "Plan my Vacation",
        "Deploy my Website",
        "Edit my Google Doc",
        "Make a Google Sheet",
        "Deploy my Website",
        "Plan my Vacation",
    ]      
    return (
      <IosOgShellCard>
        <div className="ml-auto px-4 py-2 mb-3 text-white bg-blue-500 rounded-2xl">
          <div className="text-sm md:text-base font-semibold text-base-900 truncate">
            <Typewriter texts={texts} delay={0.5} baseText="Plis " />
          </div>
        </div>
      </IosOgShellCard>
    )
}