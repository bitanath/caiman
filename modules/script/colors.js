const palette = require('image-palette')
const pixels = require('image-pixels')
const namer = require("color-namer")
const sizer = require("image-size")
const jsonfile = require("jsonfile")
const fs = require('fs')
const {HarmBlockThreshold, HarmCategory, GoogleGenerativeAI} = require("@google/generative-ai")
const generator = new GoogleGenerativeAI(process.env.GEMINI_API_KEY)

all()

async function all(){
    let files = fs.readdirSync("../data")
    for(const file of files){
        if(!/\.jpeg/.test(file)) continue
        json = jsonfile.readFileSync("../index.json")
        if(json[file]["fileName"]) continue

        if(file == "img381.jpeg") continue

        let obj = await getImageColorAlternates(`../data/${file}`)
        obj.fileName = file
        json[file] = obj
        console.log("Done",obj)
        jsonfile.writeFileSync("../index.json",json)
    }
}


async function getImageColorAlternates(filename){
    console.log("Filename",filename)
    var size = sizer(filename)
    var pix = await pixels(filename)
    var {colors} = palette(pix)
    colors = colors.map(e=>e.slice(0,-1)) //remove alpha
    let namedColors = colors.map(c=>getColorName(c))
    namedColors = namedColors.filter((value, index, self) =>
        index === self.findIndex((t) => (
          t.name === value.name
        ))
    ).map(color=>{
        let copy = color
        copy.distance = parseInt(copy.distance)
        return copy
    })
    
    let complementaryColors = await generateColorPalette(namedColors)
    namedColors = namedColors.map(color=>{
        let copy = color
        copy.hex = "#"+copy.hex
        copy.name = capitalizeFirstLetter(copy.name)
        return copy
    })
    
    let obj = {
        fileName: filename,
        imageDimensions: size,
        namedColors,
        complementaryColors
    }
    
    return obj
}

function getColorName(color){
    var name = namer(color).basic.sort((a,b)=>a.distance - b.distance)[0].name
    var tentativeNTC = namer(color).ntc.sort((a,b)=>a.distance - b.distance)[0]
    var tentativeBasic = namer(color).ntc.sort((a,b)=>a.distance - b.distance)[0]
    var hex = tentativeBasic.distance < tentativeNTC.distance ? tentativeBasic.hex.replace("#","") : tentativeNTC.hex.replace("#","")
    var distance = Math.min(tentativeBasic.distance , tentativeNTC.distance)
    return {name,hex,rgb:color,distance}
}

async function generateColorPalette(namedColors){
    console.log("Generating:")
    try{
        const gemini = generator.getGenerativeModel({ model: "gemini-1.5-flash", generationConfig: { maxOutputTokens: 100, temperature: 0.7 }, safetySettings: [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH }
        ]})
        const colors = namedColors.map(n=>n.hex)
        const prompt = `What are some complementary colours to this colour palette. Reply in JSON with an array of color name and color hex. Do not provide additional information:\n${JSON.stringify(colors)}`
        const result = await gemini.generateContent(prompt)
        const response = result.response
        const json = JSON.parse(response.text())
        let arr = []
        for(j of json){
            let vals = Object.values(j)
            let obj = {}
            if(/^#/.test(vals[0])){
                obj.name = vals[1]
                obj.hex = vals[0]
            }else if(/^#/.test(vals[1])){
                obj.name = vals[0]
                obj.hex = vals[1]
            }else{
                throw "Madar"
            }
            arr.push(obj)
        }
        
        console.log("Generated: ")
        return arr
        
    }catch(e){
        return generateColorPalette(namedColors)
    }
    
}


function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}