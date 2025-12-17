import os
import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from demo.infer import Predictor

app = FastAPI()
predictor = Predictor(
    model_arch=os.getenv("MODEL_ARCH", "maskrcnn_attfpn"),
    weights_path=os.getenv("WEIGHTS_PATH", "/weights_demo/model.pth"),
    score_thresh=float(os.getenv("SCORE_THRESH", "0.5")),
)

def _b64_png(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Karyotype demo</title></head>
      <body style="font-family: sans-serif; max-width: 1100px; margin: 20px auto;">
        <h2>Upload metaphase image</h2>
        <form action="/analyze" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*" required />
          <button type="submit">Analyze</button>
        </form>
        <p style="color:#666;">
          Output: chromosome count, XX/XY, overlay image, karyogram grid.
        </p>
      </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    res = predictor.predict(img_bytes)

    overlay_b64 = _b64_png(res["overlay_img"])
    karyo_b64 = _b64_png(res["karyogram_img"])

    return f"""
    <html>
      <head><title>Result</title></head>
      <body style="font-family: sans-serif; max-width: 1100px; margin: 20px auto;">
        <a href="/">← back</a>
        <h2>Result</h2>
        <ul>
          <li><b>Pred count:</b> {res["n_pred"]}</li>
          <li><b>Sex:</b> {res["sex"]}</li>
          <li><b>X count:</b> {res["x_count"]}, <b>Y count:</b> {res["y_count"]}</li>
        </ul>

        <h3>Overlay</h3>
        <img style="max-width: 100%; border:1px solid #ddd;" src="data:image/png;base64,{overlay_b64}" />

        <h3>Karyogram</h3>
        <img style="max-width: 100%; border:1px solid #ddd;" src="data:image/png;base64,{karyo_b64}" />
      </body>
    </html>
    """

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    res = predictor.predict(img_bytes)
    out = {
        "n_pred": res["n_pred"],
        "sex": res["sex"],
        "x_count": res["x_count"],
        "y_count": res["y_count"],
        "overlay_png_b64": _b64_png(res["overlay_img"]),
        "karyogram_png_b64": _b64_png(res["karyogram_img"]),
    }
    return JSONResponse(out)
