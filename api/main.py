from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app =FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("C:/c/code/deep_learning/Project/potato_disease/saved_models/2.h5")

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
    pass
    
    
    
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    
):
   image = read_file_as_image(await file.read())
   img_batch = np.expand_dims(image,0)
   
   predictions = MODEL.predict(img_batch)
   predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
   confidence = float(np.max(predictions[0]))
   print(predicted_class,confidence)
   return {
       "class":predicted_class,
       "confidence":confidence}
   pass
  
  
    
if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=8500)