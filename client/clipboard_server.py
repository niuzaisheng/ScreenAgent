import os
# os.environ["DISPLAY"] = ":1.0"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pyperclip
import uvicorn

SECRET_TOKEN = os.environ.get("CLIPBOARD_SERVER_SECRET_TOKEN", "secret")

app = FastAPI()

class ClipboardData(BaseModel):
    text: str
    token: str

@app.post("/clipboard")
async def update_clipboard(data: ClipboardData):
    try:
        token = data.token
        if token != SECRET_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        pyperclip.copy(data.text)
        return {"success": True, "message": "Text copied to clipboard"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
