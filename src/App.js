import { useState } from "react";

export default function PaletteGenerator() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [method, setMethod] = useState("kmeans");
  const [numColors, setNumColors] = useState(5);
  const [palette, setPalette] = useState([]);

  // Handle image selection
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  // Submit to backend
  const generatePalette = async () => {
    if (!image) {
      alert("Please upload or capture an image.");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);
    formData.append("method", method);
    formData.append("num_colors", numColors);

    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPalette(data.palette); // e.g. ["#AABBCC", "#112233"]
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div style={styles.container}>
      <h2>Color Palette Generator</h2>

      {/* Image Upload */}
      <label>
        Upload Image:
        <input type="file" accept="image/*" onChange={handleImageChange} />
      </label>

      {/* Preview */}
      {preview && <img src={preview} alt="preview" style={styles.preview} />}

      {/* Dropdown */}
      <label>
        Algorithm:
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="kmeans">Fast K-Means Original</option>
          <option value="median_cut">Fast K-Means Enhanced</option>
        </select>
      </label>

      {/* Slider */}
      <label>
        Number of Colors: {numColors}
        <input
          type="range"
          min="1"
          max="10"
          value={numColors}
          onChange={(e) => setNumColors(e.target.value)}
        />
      </label>

      {/* Generate Button */}
      <button onClick={generatePalette}>Generate Palette</button>

      {/* Palette Display */}
      <div style={styles.palette}>
        {palette.map((color, index) => (
          <div
            key={index}
            style={{ ...styles.colorBox, backgroundColor: color }}
          >
            {color}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "500px",
    margin: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  preview: {
    width: "100%",
    borderRadius: "8px",
  },
  palette: {
    display: "flex",
    gap: "8px",
    marginTop: "10px",
  },
  colorBox: {
    width: "60px",
    height: "60px",
    borderRadius: "6px",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "10px",
  },
};
