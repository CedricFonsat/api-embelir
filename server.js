import express from "express";
import cors from "cors";
import "dotenv/config";
import Replicate from "replicate";
import multer from "multer";

const app = express();
const router = express.Router();

app.use(cors());
app.use(express.static("./assets"));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(router);

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const imageBase64 = req.file.buffer.toString("base64");

    const output = await replicate.run(
      "logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df",
      {
        input: {
          pixel: "512 * 512",
          scale: 3,
          prompt: req.body.prompt || "",
          image_num: 4,
          // Utilisation de la chaîne base64 au lieu du buffer direct
          image_path: `data:image/png;base64,${imageBase64}`,
          manual_seed: -1,
          product_size: "0.5 * width",
          guidance_scale: 7.5,
          negative_prompt: "illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2)",
          num_inference_steps: 20
        }
      }
    );

    res.json({ output });
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(3000, function (err) {
  if (err) {
    console.log(err);
  } else {
    console.log(`connected to 3000`);
  }
});
