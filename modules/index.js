const bodyParser = require('body-parser')
const jsonfile = require('jsonfile')
const express = require('express');
const sharp = require('sharp')
const ejs = require('ejs')
const fs = require('fs')
const app = express()

// set the view engine to ejs
app.set('view engine', 'ejs');
app.set('views', '.')
app.use("/data",express.static('data'))
app.use(express.json({limit: '50mb'}))

app.get("/",(req,res)=>{
  res.redirect("/img1.jpeg/edit")
})

app.get('/:imagename/edit', function(req, res) {
  const json = jsonfile.readFileSync("./index.json")
  const selected = json[req.params.imagename]
  const factor = selected.imageDimensions.height > 880 || selected.imageDimensions.width > 880 ? 880.0/Math.max(selected.imageDimensions.height,selected.imageDimensions.width) : 1
  const clientWidth = factor * selected.imageDimensions.width
  const clientHeight = factor * selected.imageDimensions.height

  let namedColors
  if(selected.palette){
    namedColors = selected.palette
  }else{
    namedColors = 'The design has a color palette containing: \n'
    for(let obj of selected["namedColors"]){
      namedColors = namedColors + `The color ${obj.name} or more specifically the hex ${obj.hex}\n`
    }
  }
  
  let complementaryColors

  if(selected.complementary){
    complementaryColors = selected.complementary
  }else{
    complementaryColors = 'The design can be enhanced using complementary colors to the existing palette: \n'
    for(let obj of selected["complementaryColors"]){
      complementaryColors = complementaryColors + `The color ${obj.name} or more specifically the hex ${obj.hex}\n`
    }
  }
  
  
  const data = {
    name: `${req.params.imagename}`,
    obj: selected,
    width: clientWidth,
    height: clientHeight,
    factor,
    namedColors,
    complementaryColors
  }
  res.render('index',data);
});

app.post('/:imagename/send', function(req,res){
    //TODO parse b64 encoded image, plus the other form fields
    const received = req.body
    const json = jsonfile.readFileSync("./index.json")
    for(const key of Object.keys(received)){
      if(key == 'width' || key == 'height' || key == 'name') continue
      if(key == 'map'){
        resizeAndSaveImage(received[key],received.name,received.width,received.height)
        continue
      }
      if(received[key].length >= 1){
        json[req.params.imagename][key] = received[key]
      }
      
    }
    jsonfile.writeFileSync("./index.json",json)
    res.json({"done":"ok"})
})

app.get('/:imagename/next',function(req,res){
  const json = jsonfile.readFileSync("./index.json")
  const idx = parseInt(req.params.imagename.replace("img",""))

  const selected = `img${idx+1}.jpeg`
  if(json[selected]){
    res.redirect(`/${selected}/edit`)
  }else{
    res.redirect("/img1.jpeg/edit")
  }
})

app.listen(8080);
console.log('Server is listening on port 8080');


function resizeAndSaveImage(base64Image,name,width,height){
  let parts = base64Image.split(';');
    let mimType = parts[0].split(':')[1];
    let imageData = parts[1].split(',')[1];

    var img = Buffer.from(imageData, 'base64');
    sharp(img)
        .resize(width,height)
        .toBuffer()
        .then(resizedImageBuffer => {
            fs.writeFileSync(`gen/${name}`,resizedImageBuffer)
        })
        .catch(error => {
            // error handeling
            console.log(error)
        })
}